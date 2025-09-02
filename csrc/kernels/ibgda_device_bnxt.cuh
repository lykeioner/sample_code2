// Portions derived from NVSHMEM (https://developer.nvidia.com/nvshmem)
// Copyright (c) NVIDIA Corporation.
// Licensed under the NVSHMEM Software License Agreement (version: September 3, 2019).
// See full license at: https://docs.nvidia.com/nvshmem/api/sla.html
//
// Modified from original source:
//  - nvshmem/src/include/non_abi/device/pt-to-pt/ibgda_device.cuh
#pragma once

#include "configs.cuh"
#include "exception.cuh"
#include "utils.cuh"
#include "infiniband/bnxt_re_hsi.h"

#undef htole16
#define htole16(x)        (x)

#undef htole32
#define htole32(x)        (x)

#undef htobe32
#define htobe32(x)        (x)

#undef htole64
#define htole64(x)        (x)

#undef le32toh
#define le32toh(x)        (x)

namespace deep_ep {

EP_STATIC_ASSERT(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64, "Invalid QP minimum depth");

__device__ static __forceinline__
uint64_t HtoBE64(uint64_t x) {
    uint64_t ret;
    asm("{\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 lo;\n\t"
        ".reg .b32 hi;\n\t"
        ".reg .b32 new_lo;\n\t"
        ".reg .b32 new_hi;\n\t"
        "mov.b64 {lo,hi}, %1;\n\t"
        "prmt.b32 new_hi, lo, ign, 0x0123;\n\t"
        "prmt.b32 new_lo, hi, ign, 0x0123;\n\t"
        "mov.b64 %0, {new_lo,new_hi};\n\t"
        "}" : "=l"(ret) : "l"(x));
    return ret;
}

__device__ static __forceinline__
uint32_t HtoBE32(uint32_t x) {
    uint32_t ret;
    asm("{\n\t"
        ".reg .b32 ign;\n\t"
        "prmt.b32 %0, %1, ign, 0x0123;\n\t"
        "}" : "=r"(ret) : "r"(x));
    return ret;
}

__device__ static __forceinline__
uint16_t HtoBE16(uint16_t x) {
    // TODO: simplify PTX using 16-bit instructions
    auto a = static_cast<uint32_t>(x);
    uint32_t d;
    asm volatile(
        "{\n\t"
        ".reg .b32 mask;\n\t"
        ".reg .b32 ign;\n\t"
        "mov.b32 mask, 0x4401;\n\t"
        "mov.b32 ign, 0x0;\n\t"
        "prmt.b32 %0, %1, ign, mask;\n\t"
        "}"
        : "=r"(d)
        : "r"(a));
    return static_cast<uint16_t>(d);
}

typedef struct bnxt_re_bsqe __attribute__((__aligned__(8))) ibgda_bnxt_ctrl_seg_t;

#define bnxt_re_get_cqe_sz()    (sizeof(struct bnxt_re_req_cqe) +   \
                 sizeof(struct bnxt_re_bcqe))
#define bnxt_re_is_cqe_valid(valid, phase)              \
                (((valid) & BNXT_RE_BCQE_PH_MASK) == (phase))

typedef struct {
    uint32_t add_data;
    uint32_t field_boundary;
    uint64_t reserved;
} __attribute__((__packed__)) ibgda_atomic_32_masked_fa_seg_t;

__device__ static __forceinline__
nvshmemi_ibgda_device_state_t* ibgda_get_state() {
    return &nvshmemi_ibgda_device_state_d;
}

__device__ static __forceinline__
nvshmemi_ibgda_device_qp_t* ibgda_get_rc(int pe, int id) {
    auto state = ibgda_get_state();
    const auto num_rc_per_pe = ibgda_get_state()->num_rc_per_pe;
    return &state->globalmem.rcs[pe * num_rc_per_pe + id % num_rc_per_pe];
}

__device__ static __forceinline__
void ibgda_lock_acquire(int *lock) {
    while (atomicCAS(lock, 0, 1) == 1);

    // Prevent reordering before the lock is acquired
    memory_fence_cta();
}

__device__ static __forceinline__
void ibgda_lock_release(int *lock) {
    memory_fence_cta();

    // Prevent reordering before lock is released
    st_na_relaxed(lock, 0);
}

__device__ void static *bnxt_re_pull_psn_buff(nvshmemi_ibgda_device_qp_t *qp,
                uint32_t msn_idx) {
   // MSN entries are 64b wide.
   return (void *)(((char *) qp->pad) + (msn_idx << 3));
}

__device__ uint64_t static bnxt_re_update_msn_tbl(uint32_t st_idx, uint32_t npsn, uint32_t start_psn) {
   return ((((uint64_t)(st_idx) << BNXT_RE_SQ_MSN_SEARCH_START_IDX_SHIFT) &
                       BNXT_RE_SQ_MSN_SEARCH_START_IDX_MASK) |
                       (((uint64_t)(npsn) << BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_SHIFT) &
                       BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_MASK) |
                       (((start_psn) << BNXT_RE_SQ_MSN_SEARCH_START_PSN_SHIFT) &
                       BNXT_RE_SQ_MSN_SEARCH_START_PSN_MASK));
}
__device__ void static  bnxt_re_fill_psns_for_msntbl(
    nvshmemi_ibgda_device_qp_t *qp, uint16_t slot_idx, uint32_t msn_idx, uint32_t start_psn,
    uint32_t pkt_cnt) {
    struct bnxt_re_msns msns;
    uint64_t *msns_ptr;

    msns_ptr = (uint64_t *)bnxt_re_pull_psn_buff(qp, msn_idx);
    msns.start_idx_next_psn_start_psn =
                        bnxt_re_update_msn_tbl(slot_idx, start_psn + pkt_cnt, start_psn);

   st_na_release(msns_ptr, *(reinterpret_cast<uint64_t*>(&msns)));
}

// This is a simplified version of NVSHMEM's `ibgda_poll_cq`. 
// Note that this implementation does not guarantee thread safety,
// so we must ensure that no other threads are concurrently using the same QP.
__device__ static __forceinline__ void
ibgda_poll_cq(nvshmemi_ibgda_device_cq_t *cq, uint64_t idx) {
    memory_fence_cta();

    struct bnxt_re_req_cqe *hwcqe = (struct bnxt_re_req_cqe *)cq->cqe;
    bnxt_re_bcqe *hdr;
    uint32_t flg_val;

    auto cons_idx = ld_na_relaxed(cq->cons_idx);
#ifdef NVSHMEM_IBGDA_DEBUG
    bool valid_comp = false;
#endif
    uint8_t cqe_status = 0;
    
    // If idx is a lot greater than cons_idx, we might get incorrect result due
    // to wqe_counter wraparound. We need to check prod_idx to be sure that idx
    // has already been submitted.
    while (ld_na_relaxed(cq->prod_idx) < idx)
        ;
    memory_fence_cta();

    ibgda_lock_acquire(cq->poll_cq_lock);

    cons_idx = ld_na_relaxed(cq->cons_idx);
    auto prod_idx = ld_na_relaxed(cq->prod_idx);
    auto cqe_idx = ld_na_relaxed(cq->cqe_idx);
    if (idx <= cons_idx)
        goto poll_done;

    // Handle some CQ polling TBD. CQ poll might be called for periodic
    // check and it is possible to have no potential completion on that CQ.
    // Currently it is handled through timeout.
    do {
        cons_idx = ld_na_relaxed(cq->cons_idx);
        prod_idx = ld_na_relaxed(cq->prod_idx);
        cqe_idx = ld_na_relaxed(cq->cqe_idx);

        hwcqe = (struct bnxt_re_req_cqe *)((unsigned long)cq->cqe +
                 (cqe_idx * bnxt_re_get_cqe_sz()));
        hdr = (bnxt_re_bcqe*)((unsigned long)hwcqe + sizeof(bnxt_re_req_cqe));
        flg_val = le32toh(hdr->flg_st_typ_ph);

#ifdef NVSHMEM_IBGDA_DEBUG
        uint32_t *cqe_slot;
        int i;

        cqe_slot = (uint32_t *)(uint32_t*)hwcqe;
        for (i = 0; i < 1; i++) {
            printf("DEEP_EP: hwcqe 0x%lx : %08x %08x %08x %08x (qpn 0x%x slot %ld)\n",
                 &cqe_slot[0], (cqe_slot[1]), (cqe_slot[0]),
                 (cqe_slot[3]), (cqe_slot[2]), cq->qpn, cqe_idx + i);
            cqe_slot = cqe_slot + 4;
        }
        printf("DEEP_EP: qpn 0x%x  flg_val %08x  ready_head 0x%lx resv_head 0x%lx"
            "idx from caller 0x%lx tx prod 0x%lx tx cons 0x%lx/0x%lx "
            "(hw sq_cons 0x%lx) phase 0x%lx\n",
            cq->qpn, flg_val,
            ld_na_relaxed(cq->ready_head), ld_na_relaxed(cq->resv_head), idx,
            ld_na_relaxed(cq->prod_idx), cons_idx, cqe_idx,
            ld_na_relaxed(cq->sq_cons_idx), ld_na_relaxed(cq->cq_phase));
#endif
        if (bnxt_re_is_cqe_valid(flg_val, (uint32_t)ld_na_relaxed(cq->cq_phase))) {
            cqe_idx = (cqe_idx + 1) % cq->ncqes;
            if (cqe_idx == 0)
                atomicXor(reinterpret_cast<unsigned long long*>(cq->cq_phase), 0x1);

            atomicExch(reinterpret_cast<unsigned long long*>(cq->cqe_idx), cqe_idx);

            int wqe_cnt = hwcqe->con_indx - (int)ld_na_relaxed(cq->sq_cons_idx);
            if (wqe_cnt < 0)
                wqe_cnt += cq->sq_size;
            cons_idx += wqe_cnt;
            atomicMax(reinterpret_cast<unsigned long long*>(cq->cons_idx), cons_idx);
            atomicExch(reinterpret_cast<unsigned long long*>(cq->sq_cons_idx),
                            (unsigned long long int)hwcqe->con_indx);

            cqe_status = (flg_val >> BNXT_RE_BCQE_STATUS_SHIFT) &
                          BNXT_RE_BCQE_STATUS_MASK;

            if (cqe_status) {
                goto check_opcode;
            }
        }
    } while (cons_idx < idx);

check_opcode:
    /* TBD CQE_REQ_ERR Case handling*/

#ifdef NVSHMEM_IBGDA_DEBUG
    printf(
        "[%d] DEEP_EP: ibgda_poll_cq %s: \n"
        "    cons_idx=%#lx, prod_idx=%#lx, cqn=%#x, qpn=%#x \n"
        "    resv_head=%#lx, ready_head=%#lx\n"
        "    while waiting for idx=%#lx. cqe_status 0x%x\n",
        nvshmemi_device_state_d.mype, valid_comp ? "SUCESS" : "TIMEOUT",
        cons_idx, ld_na_relaxed(cq->prod_idx), cq->cqn, cq->qpn,
        ld_na_relaxed(cq->resv_head), ld_na_relaxed(cq->ready_head), idx,
        cqe_status);
#endif

poll_done:
    // Prevent reordering of this function and later instructions
    memory_fence_cta();
    ibgda_lock_release(cq->poll_cq_lock);
}

// Updates the last slot idx unconditionally with wrap consideration
__device__ static __forceinline__ uint64_t ibgda_reserve_slot_idx(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t num_slots) {
    unsigned long long int *addr, oldval, assumed, newval;

    addr = (unsigned long long int *)&qp->mvars.tx_wq.resv_prod_slot_idx;
    if (!num_slots)
        return atomicAdd(addr, 0);  //Safe read

    oldval = atomicAdd(addr, 0);  //First read
    do {
        assumed = oldval;
        newval = (assumed + num_slots) % qp->tx_wq.sq_depth;
        oldval = atomicCAS(addr, assumed, newval);
    } while (oldval != assumed);

#ifdef NVSHMEM_IBGDA_DEBUG
    printf("[%d] update slot idx from %lld to %lld\n",
           nvshmemi_device_state_d.mype, oldval, newval);
#endif

    return oldval;
}

__device__ static __forceinline__ uint64_t bnxt_re_get_pkts_per_wqe(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t data_bytes) {
    if (!data_bytes)
        data_bytes = 1;
    return (data_bytes + qp->mtu - 1) / qp->mtu;
}

// Wait until wqe `idx - 1` is completed.
__device__ static __forceinline__ void
nvshmemi_ibgda_quiet(int dst_pe, int qp_id) {
    auto qp = ibgda_get_rc(dst_pe, qp_id);
    uint64_t prod_idx = ld_na_relaxed(qp->tx_wq.prod_idx);
    ibgda_poll_cq(qp->tx_wq.cq, prod_idx);
}

// Helper routine to provide a byte pointer of the TX HWQ (input = WQE idx)
__device__ static __forceinline__ void*
ibgda_get_wqe_slot_ptr(nvshmemi_ibgda_device_qp_t* qp, uint64_t slot_idx) {
    slot_idx = slot_idx % qp->tx_wq.sq_depth;
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(qp->tx_wq.wqe) +
                    + (slot_idx * BNXT_RE_SLOT_SIZE_BB));
}

// Allow the number of slots for RQ to be different from the SQ
__device__ static __forceinline__ void*
ibgda_get_rqe_ptr(nvshmemi_ibgda_device_qp_t* qp, uint16_t rqe_idx) {
    uint16_t cnt = qp->rx_wq.nwqes;
    uint16_t idx = rqe_idx & (cnt - 1);
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(qp->rx_wq.wqe) + (idx << BNXT_RE_STATIC_RQE_SHIFT));
}


__device__ void static bnxt_re_init_db_hdr(struct bnxt_re_db_hdr *hdr,
       uint32_t indx, uint32_t epoch, uint32_t qid, uint32_t typ) {
   uint64_t key_lo, key_hi;

   key_lo = htole32((indx & BNXT_RE_DB_INDX_MASK) |
                    (epoch << BNXT_RE_DB_EPOCH_SHIFT));
   key_hi = htole32((qid & BNXT_RE_DB_QID_MASK) |
            ((typ & BNXT_RE_DB_TYP_MASK) << BNXT_RE_DB_TYP_SHIFT) |
            (0x1UL << BNXT_RE_DB_VALID_SHIFT));
   hdr->typ_qid_indx = htole64((key_lo | (key_hi << 32)));
}

__device__ static __forceinline__
void ibgda_ring_recv_db(nvshmemi_ibgda_device_qp_t *qp, uint16_t prod_idx) {
    // Assumes the rx_wq.bf is the GPU VA that is mapped to the DPI
    uint32_t epoch = (uint32_t)ld_na_relaxed(&qp->mvars.rx_wq.epoch);
    uint64_t *bf_ptr = (uint64_t *)qp->rx_wq.bf;
    struct bnxt_re_db_hdr hdr;

    // Do no convert prod_idx to slot for RQ.
    bnxt_re_init_db_hdr(&hdr, prod_idx, epoch, qp->qpn, BNXT_RE_QUE_TYPE_RQ);
    
#ifdef NVSHMEM_IBGDA_DEBUG
    printf("DEEP_EP: From %s %d qpn 0x%x prod_idx %#x at 0x%lx cq_handle 0x%lx\n",
                    __func__, __LINE__, qp->qpn, prod_idx,
                    (unsigned long)bf_ptr, (unsigned long)qp->rx_wq.cq);
#endif

    st_na_release(bf_ptr, *(reinterpret_cast<uint64_t*>(&hdr)));
    
#ifdef NVSHMEM_IBGDA_DEBUG
    uint32_t *dst = (uint32_t *)&hdr;
    printf("DEEP_EP: recv_db: %08x %08x qpn 0x%x\n", (dst[1]), (dst[0]), qp->qpn); 
#endif
}

__device__ __forceinline__
void ibgda_post_recv(nvshmemi_ibgda_device_qp_t *qp, uint64_t new_prod_idx) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t old_prod_idx = 0;

    // Comes here from allocate_recvs which is a single thread operation, skip the lock
    // TBD - Remove the lock OR use post_recv_lock.
    ibgda_lock_acquire(&mvars->post_send_lock);
    
    if (new_prod_idx > old_prod_idx) {
        ibgda_ring_recv_db(qp, new_prod_idx);
    }
    ibgda_lock_release(&mvars->post_send_lock);
}
__device__ static __forceinline__
void ibgda_ring_db(nvshmemi_ibgda_device_qp_t *qp, uint64_t slot_idx) {
    uint32_t prod_slot_idx = (uint32_t)slot_idx;
    uint32_t epoch = (uint32_t)ld_na_relaxed(&qp->mvars.tx_wq.epoch);
    uint64_t *bf_ptr = (uint64_t *)qp->tx_wq.bf;
    struct bnxt_re_db_hdr hdr;
   
    bnxt_re_init_db_hdr(&hdr, prod_slot_idx, epoch, qp->qpn, BNXT_RE_QUE_TYPE_SQ);

#ifdef NVSHMEM_IBGDA_DEBUG
    printf("DEEP_EP: From %s %d qpn 0x%x  slot_idx 0x%lx cq_handle 0x%lx\n",
           __func__, __LINE__, qp->qpn,
          (unsigned long)prod_slot_idx, (unsigned long)qp->tx_wq.cq);
#endif
    st_na_release(bf_ptr, *(reinterpret_cast<uint64_t*>(&hdr)));
}

__device__ static __forceinline__
void ibgda_post_send(nvshmemi_ibgda_device_qp_t *qp, uint64_t new_prod_idx,
                uint64_t new_slot_idx) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t old_prod_idx;

    // Update `prod_idx` before ringing the doorbell, so that we know which index is needed in quiet/fence
    ibgda_lock_acquire(&mvars->post_send_lock);

    old_prod_idx = atomicMax(reinterpret_cast<unsigned long long int*>(&mvars->tx_wq.prod_idx), new_prod_idx);
    if (new_prod_idx > old_prod_idx) {
        if (new_slot_idx <
            atomicAdd(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.posted_prod_slot_idx), 0ULL))
            atomicXor(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.epoch), 0x1ULL);

        atomicExch(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.posted_prod_slot_idx),
                       (unsigned long long int)new_slot_idx);
        ibgda_ring_db(qp, new_slot_idx);
    }

    ibgda_lock_release(&mvars->post_send_lock);
}

template <bool kAlwaysDoPostSend>
__device__ static __forceinline__
void ibgda_submit_requests(nvshmemi_ibgda_device_qp_t *qp, uint64_t base_wqe_idx,
                           uint32_t num_wqes,
                           uint64_t base_slot_idx, uint64_t num_slots,
                           int message_idx = 0) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t new_wqe_idx = base_wqe_idx + num_wqes;
    uint64_t new_slot_idx = (base_slot_idx + num_slots) % qp->tx_wq.sq_depth;

    // WQE writes must be finished first
    __threadfence();

    // Wait for prior WQE slots to be filled first
    auto *ready_idx = reinterpret_cast<unsigned long long int*>(&mvars->tx_wq.ready_head);
    auto *ready_slot = reinterpret_cast<unsigned long long int*>(&mvars->tx_wq.ready_prod_slot_idx);
    while (atomicCAS(ready_idx, base_wqe_idx, new_wqe_idx) != base_wqe_idx);
    while (atomicCAS(ready_slot, base_slot_idx, new_slot_idx) != base_slot_idx);

    // Always post, not in batch
    constexpr int kNumRequestInBatch = 4;
    if (kAlwaysDoPostSend or (message_idx + 1) % kNumRequestInBatch == 0)
        ibgda_post_send(qp, new_wqe_idx, new_slot_idx);
}

__device__ static __forceinline__ void
ibgda_write_rdma_write_inl_wqe(nvshmemi_ibgda_device_qp_t *qp, const uint32_t *val, uint64_t raddr,
                               __be32 rkey, uint32_t bytes, uint64_t wqe_slot_idx,
                               uint64_t msn_idx, uint64_t psn,
                               uint32_t imm) {
#ifdef NVSHMEM_IBGDA_DEBUG
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint32_t *dst;
#endif
    ibgda_bnxt_ctrl_seg_t ctrl_seg;
    struct bnxt_re_rdma raddr_seg;
    uint32_t slots = 2;
   
    ibgda_bnxt_ctrl_seg_t *ctrl_seg_ptr = (ibgda_bnxt_ctrl_seg_t *)ibgda_get_wqe_slot_ptr(qp, wqe_slot_idx);
    struct bnxt_re_rdma *raddr_seg_ptr = (struct bnxt_re_rdma *)ibgda_get_wqe_slot_ptr(qp, wqe_slot_idx + 1);
    struct bnxt_re_sge *wqe_data_seg_ptr = (struct bnxt_re_sge *)ibgda_get_wqe_slot_ptr(qp, wqe_slot_idx + 2);

    slots += (bytes + (BNXT_RE_SLOT_SIZE_BB - 1)) / BNXT_RE_SLOT_SIZE_BB;

    ctrl_seg.rsv_ws_fl_wt = htole32((slots << BNXT_RE_HDR_WS_SHIFT) |
                                (BNXT_RE_WR_FLAGS_SIGNALED << BNXT_RE_HDR_FLAGS_SHIFT) |
                                (BNXT_RE_WR_FLAGS_UC_FENCE << BNXT_RE_HDR_FLAGS_SHIFT) |
                                (BNXT_RE_WR_FLAGS_INLINE << BNXT_RE_HDR_FLAGS_SHIFT) |
                                BNXT_RE_WR_OPCD_RDMA_WRITE_IMM);
    ctrl_seg.key_immd = htole32(imm);
    ctrl_seg.lhdr.qkey_len = bytes;

    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16, "sizeof(*ctrl_seg_ptr) == 16");
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));

#ifdef NVSHMEM_IBGDA_DEBUG
    dst = (uint32_t *)ctrl_seg_ptr;
    printf("DEEP_EP slot 0: %08x %08x ctrl_seg_ptr 0x%lx qpn 0x%x val 0x%x bytes 0x%x"
           " resv %#lx sq_cons_idx %#lx\n",
            (dst[1]), (dst[0]), (unsigned long) ctrl_seg_ptr, qp->qpn, *val,
            bytes, ld_na_relaxed(&mvars->tx_wq.resv_head),
            ld_na_relaxed(&mvars->tx_wq.sq_cons_idx));
    printf("      : %08x %08x\n", (dst[3]), (dst[2]));
#endif
    
    raddr_seg.rva = htole64(raddr);
    raddr_seg.rkey = HtoBE32(rkey);
    raddr_seg.ts = 0;
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));

#ifdef NVSHMEM_IBGDA_DEBUG
    dst = (uint32_t *)raddr_seg_ptr;
    printf("DEEP_EP: slot 1: %08x %08x\n", (dst[1]), (dst[0]));
    printf("       : %08x %08x\n", (dst[3]), (dst[2]));
#endif
    st_na_relaxed(reinterpret_cast<uint32_t*>(wqe_data_seg_ptr), *reinterpret_cast<const uint32_t*>(val));
#ifdef NVSHMEM_IBGDA_DEBUG
    dst = (uint32_t *)wqe_data_seg_ptr;
    printf("DEEP_EP: slot 2: %08x %08x\n", (dst[1]), (dst[0]));
    printf("       : %08x %08x\n", (dst[3]), (dst[2]));
#endif

    // Calculate and fill start and end PSN of the WQE
    bnxt_re_fill_psns_for_msntbl(qp, wqe_slot_idx, msn_idx, psn, 1);
}

__device__ static __forceinline__
uint64_t ibgda_get_lkey_and_rkey(uint64_t laddr, __be32 *lkey,
                                 uint64_t raddr, int dst_pe, uint64_t *out_raddr, __be32 *out_rkey) {
    auto state = ibgda_get_state();
    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);
    auto log2_cumem_granularity = state->log2_cumem_granularity;

    // Local key
    uint64_t idx = (laddr - heap_start) >> log2_cumem_granularity;
    auto device_key = state->constmem.lkeys[idx];
    auto lchunk_size = device_key.next_addr - laddr;
    *lkey = device_key.key;

    // Remote key
    uint64_t roffset = raddr - heap_start;
    idx = ((roffset >> log2_cumem_granularity) * nvshmemi_device_state_d.npes) + dst_pe;
    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS) {
        device_key = state->constmem.rkeys[idx];
    } else {
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];
    }
    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;
    *out_rkey = device_key.key;

    // Return the minimum of local and remote chunk sizes
    auto rchunk_size = device_key.next_addr - roffset;
    return min(lchunk_size, rchunk_size);
}

__device__ static __forceinline__ void
ibgda_get_rkey(uint64_t addr, int dst_pe, uint64_t *out_raddr, __be32 *out_rkey) {
    auto state = ibgda_get_state();
    auto heap_start = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);

    uint64_t roffset = addr - heap_start;
    uint64_t idx = ((roffset >> state->log2_cumem_granularity) * nvshmemi_device_state_d.npes) + dst_pe;
    nvshmemi_ibgda_device_key_t device_key;
    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS)
        device_key = state->constmem.rkeys[idx];
    else
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];
    *out_raddr = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.peer_heap_base_remote[dst_pe]) + roffset;
    *out_rkey = device_key.key;
}


__device__ static __forceinline__ uint64_t
ibgda_reserve_wqe_slots(nvshmemi_ibgda_device_qp_t *qp, uint32_t num_wqes,
                int wqe_size, int num_msn, int num_pkts, uint64_t *slot_idx,
                uint64_t *msn, uint64_t *psn) {
    auto mvars = &qp->mvars;
    uint64_t wqe_idx;

    ibgda_lock_acquire(&qp->mvars.resv_lock);
    // 1. Reserve wqe_idx
    wqe_idx = atomicAdd(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.resv_head), num_wqes);

    // 2. Reserve slot_idx
    *slot_idx = atomicAdd(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.resv_prod_slot_idx), 0);
    atomicExch(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.resv_prod_slot_idx),
               (*slot_idx + num_wqes * wqe_size) % qp->tx_wq.sq_depth);

    // 3. Reserve msn and psn idx
    *msn = atomicAdd(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.msn_idx), 0);
    atomicExch(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.msn_idx),
               (*msn + num_msn) % qp->msn_tbl_sz);
    *psn = atomicAdd(reinterpret_cast<unsigned long long*>(&mvars->tx_wq.psn), num_pkts);

    ibgda_lock_release(&qp->mvars.resv_lock);

    // If last slot is available, all prior slots are also available.
    //ibgda_wait_for_slot_availability(qp, wqe_idx + num_wqes);
    return wqe_idx;
}

// CQE prod/cons advancement routines (input = CQE idx, output byte ptr)
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void *ibgda_get_cqe_ptr(
    nvshmemi_ibgda_device_cq_t *cq, uint16_t cqe_idx) {
    uint16_t cnt = cq->ncqes;
    uint16_t idx = cqe_idx & (cnt - 1);
    return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(cq->cqe) + (idx << BNXT_RE_STATIC_CQE_SHIFT));
}

// Wait for one completion and quit.
// We post recv and poll recv in the same thread, so we don't need to maintain queue status.
__device__ static __forceinline__ void
nvshmemi_ibgda_poll_recv(int dst_pe, int qp_id) {

    memory_fence_cta();

    auto qp = ibgda_get_rc(dst_pe, qp_id);
    auto cq = qp->rx_wq.cq;

    bnxt_re_bcqe *hdr;
    uint32_t flg_val;
    struct bnxt_re_req_cqe *hwcqe = (struct bnxt_re_req_cqe *)cq->cqe;

    auto cqe_idx = ld_na_relaxed(cq->cqe_idx);
    
    do {
        cqe_idx = ld_na_relaxed(cq->cqe_idx);

        hwcqe = (struct bnxt_re_req_cqe *)((unsigned long)cq->cqe +
                 (cqe_idx * bnxt_re_get_cqe_sz()));
        hdr = (bnxt_re_bcqe*)((unsigned long)hwcqe + sizeof(bnxt_re_req_cqe));
        flg_val = le32toh(ld_na_relaxed(&hdr->flg_st_typ_ph));

#ifdef NVSHMEM_IBGDA_DEBUG
        uint32_t *cqe_slot;
        int i;
        cqe_slot = (uint32_t *)(uint32_t*)hwcqe;
        for (i = 0; i < 2; i++) {
            printf(">>> DEEP_EP: hwcqe 0x%lx : %08x %08x %08x %08x (qpn 0x%x slot %ld)\n",
                    &cqe_slot[0], (cqe_slot[1]), (cqe_slot[0]),
                    (cqe_slot[3]), (cqe_slot[2]), cq->qpn, cqe_idx + i);
                    cqe_slot = cqe_slot + 4;
        }
#endif

        // One poll one completion is good enough for now.
        // TBD - If required add additional logic.
        if (bnxt_re_is_cqe_valid(flg_val, ld_na_relaxed(cq->cq_phase))) {
            cqe_idx = (cqe_idx + 1) % cq->ncqes;
            if (cqe_idx == 0)
                atomicXor(reinterpret_cast<unsigned long long*>(cq->cq_phase), 0x1);

            atomicExch(reinterpret_cast<unsigned long long*>(cq->cqe_idx), cqe_idx);
            atomicAdd(reinterpret_cast<unsigned long long*>(cq->cons_idx), 0x1);
            break;
        }
    } while (1);

    // Prevent reordering of this function and later instructions
    memory_fence_cta();
}

__device__ static __forceinline__ void
nvshmemi_ibgda_rma_p(int *rptr, const int value, int dst_pe, int qp_id, uint32_t imm = std::numeric_limits<uint32_t>::max()) {
    uint64_t base_wqe_idx = 0, base_slot_idx = 0, base_msn_idx = 0, base_psn = 0;
    int total_msn, ppw, total_pkts;
    uint64_t transfer_size = 4; // Inline data size
    int num_slots_per_wqe = 3;
    int num_wqes = 1;

    // Get rkey
    // NOTES: the `p` operation will not cross multiple remote chunks
    __be32 rkey;
    uint64_t raddr;

    ibgda_get_rkey(reinterpret_cast<uint64_t>(rptr), dst_pe, &raddr, &rkey);

    // Write WQEs
    auto qp = ibgda_get_rc(dst_pe, qp_id);

    total_msn = num_wqes;
    ppw = bnxt_re_get_pkts_per_wqe(qp, transfer_size);
    total_pkts = total_msn * ppw;

    base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes,
                                           num_slots_per_wqe, total_msn, total_pkts,
                                           &base_slot_idx, &base_msn_idx, &base_psn);

    ibgda_write_rdma_write_inl_wqe(qp, reinterpret_cast<const uint32_t*>(&value), raddr, rkey,
                    transfer_size, base_slot_idx, base_msn_idx, base_psn, imm);

    // Submit requests
    ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes, base_slot_idx,
                    num_wqes * num_slots_per_wqe);
}

__device__ static __forceinline__ void
ibgda_write_rdma_write_wqe(nvshmemi_ibgda_device_qp_t *qp, uint64_t laddr, __be32 lkey,
                           uint64_t raddr, __be32 rkey, uint32_t bytes,
                           uint64_t wqe_slot_idx, uint64_t msn_idx, uint64_t psn,
                           uint32_t npkts) {
#ifdef NVSHMEM_IBGDA_DEBUG
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint32_t *dst;
#endif
    ibgda_bnxt_ctrl_seg_t ctrl_seg;
    struct bnxt_re_rdma raddr_seg;
    struct bnxt_re_sge data_seg;
    uint32_t slots = 3;
    // All segments are within the same WQE
    ibgda_bnxt_ctrl_seg_t *ctrl_seg_ptr = (ibgda_bnxt_ctrl_seg_t *)ibgda_get_wqe_slot_ptr(qp, wqe_slot_idx);
    struct bnxt_re_rdma *raddr_seg_ptr = (struct bnxt_re_rdma *)ibgda_get_wqe_slot_ptr(qp, wqe_slot_idx + 1);
    struct bnxt_re_sge *data_seg_ptr = (struct bnxt_re_sge *)ibgda_get_wqe_slot_ptr(qp, wqe_slot_idx + 2);

    ctrl_seg = { 0 };
    ctrl_seg.rsv_ws_fl_wt = htole32((slots << BNXT_RE_HDR_WS_SHIFT) |
                                (BNXT_RE_WR_FLAGS_SIGNALED << BNXT_RE_HDR_FLAGS_SHIFT) |
                                    BNXT_RE_WR_OPCD_RDMA_WRITE);
    ctrl_seg.key_immd = 0;
    ctrl_seg.lhdr.qkey_len = htole32(bytes);

    raddr_seg.rva = htole64(raddr);
    raddr_seg.rkey = HtoBE32(rkey);
    raddr_seg.ts = 0;

    data_seg.length = htole32(bytes);
    data_seg.lkey = HtoBE32(lkey);
    data_seg.pa = htole64(laddr);

    EP_STATIC_ASSERT(sizeof(*ctrl_seg_ptr) == 16, "sizeof(*ctrl_seg_ptr) == 16");
    st_na_relaxed(reinterpret_cast<int4*>(ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));
#ifdef NVSHMEM_IBGDA_DEBUG
    dst = (uint32_t *)ctrl_seg_ptr;
    printf("DEEP_EP slot 0: %08x %08x ctrl_seg_ptr 0x%lx qpn 0x%x bytes 0x%x"
           " resv %#lx sq_cons_idx %#lx\n",
            (dst[1]), (dst[0]), (unsigned long) ctrl_seg_ptr, qp->qpn,
            bytes, ld_na_relaxed(&mvars->tx_wq.resv_head),
            ld_na_relaxed(&mvars->tx_wq.sq_cons_idx));
    printf("       : %08x %08x\n", (dst[3]), (dst[2]));
#endif

    EP_STATIC_ASSERT(sizeof(*raddr_seg_ptr) == 16, "sizeof(*raddr_seg_ptr) == 16");
    st_na_relaxed(reinterpret_cast<int4*>(raddr_seg_ptr), *reinterpret_cast<const int4*>(&raddr_seg));
#ifdef NVSHMEM_IBGDA_DEBUG
    dst = (uint32_t *)raddr_seg_ptr;
    printf("DEEP_EP slot 1: %08x %08x\n", (dst[1]), (dst[0]));
    printf("      : %08x %08x\n", (dst[3]), (dst[2]));
#endif
    EP_STATIC_ASSERT(sizeof(*data_seg_ptr) == 16, "sizeof(*data_seg_ptr) == 16");
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<const int4*>(&data_seg));
#ifdef NVSHMEM_IBGDA_DEBUG
    dst = (uint32_t *)data_seg_ptr;
    printf("DEEP_EP slot 2: %08x %08x\n", (dst[1]), (dst[0]));
    printf("      : %08x %08x\n", (dst[3]), (dst[2]));
#endif

    // Calculate and fill start and end PSN of the WQE
    bnxt_re_fill_psns_for_msntbl(qp, wqe_slot_idx, msn_idx, psn, npkts);
}

// This is used for clean up only; not sure if its needed for Thor2
__device__ static __forceinline__ void
ibgda_write_empty_recv_wqe(void *out_wqe) {
    auto *rq_ctrl_seg_ptr = reinterpret_cast<struct bnxt_re_brqe *>(out_wqe);
    auto resv_seg_ptr = reinterpret_cast<struct bnxt_re_rdma *>(reinterpret_cast<uintptr_t>(rq_ctrl_seg_ptr) + sizeof(*rq_ctrl_seg_ptr));
    auto data_seg_ptr = reinterpret_cast<struct bnxt_re_sge *>(reinterpret_cast<uintptr_t>(resv_seg_ptr) + sizeof(*resv_seg_ptr));
    struct bnxt_re_sge data_seg;
    struct bnxt_re_brqe ctrl_seg;

    ctrl_seg.rsv_ws_fl_wt = htole32((BNXT_RE_STATIC_WQE_SIZE_SLOTS << BNXT_RE_HDR_WS_SHIFT) |
                                BNXT_RE_WR_OPCD_RECV);
    ctrl_seg.wrid = 0xFF;
    st_na_relaxed(reinterpret_cast<int4*>(rq_ctrl_seg_ptr), *reinterpret_cast<const int4*>(&ctrl_seg));
    // Make the first segment in the WQE invalid, then the entire list will be invalid
    data_seg.length = 0;
    data_seg.lkey = htole64(0xFFFF);
    data_seg.pa = 0;

    EP_STATIC_ASSERT(sizeof(data_seg) == sizeof(int4), "Invalid data type length");
    st_na_relaxed(reinterpret_cast<int4*>(data_seg_ptr), *reinterpret_cast<const int4*>(&data_seg));
}

__device__ static __forceinline__ uint64_t
nvshmemi_ibgda_allocate_recvs(nvshmemi_ibgda_device_qp* qp) {
    auto mvars = &qp->mvars;
    uint64_t wrap_count;

    // Allocate if not enough
    constexpr int kMinIBGDARecvs = 32;
    auto resv_head = mvars->rx_wq.resv_head;
    auto num_valid_slots = resv_head - mvars->rx_wq.cons_idx;
    if (num_valid_slots < kMinIBGDARecvs) {
        resv_head = ld_na_relaxed(&mvars->rx_wq.cons_idx) + qp->rx_wq.nwqes - num_valid_slots;
#ifdef NVSHMEM_IBGDA_DEBUG
        printf("DEEP_EP %s resv_head 0x%lx 0x%lx cons_idx 0x%lx\n", __func__,
                        ld_na_relaxed(&mvars->rx_wq.resv_head),
                        resv_head,
                        ld_na_relaxed(&mvars->rx_wq.cons_idx));
#endif
        mvars->rx_wq.resv_head = resv_head;
        wrap_count = (resv_head - 1) / qp->rx_wq.nwqes;
        if ((wrap_count % 2) != ld_na_relaxed(&mvars->rx_wq.epoch))
            atomicXor(reinterpret_cast<unsigned long long*>(&mvars->rx_wq.epoch), 0x1ULL);
        ibgda_post_recv(qp, htole32(((resv_head - 1) % qp->rx_wq.nwqes) & 0xffff));
    }

    // Return old number of slots
    return num_valid_slots;
}

__device__ static __forceinline__ void
nvshmemi_ibgda_prepare_recvs(int dst_rank, int qp_id) {
    // NOTES: only one thread can run this function
    // TODO: consider this assertion for normal AR
    EP_DEVICE_ASSERT(nvshmemi_ibgda_allocate_recvs(ibgda_get_rc(dst_rank, qp_id)) > 16);
}

template <bool kAlwaysDoPostSend = false>
__device__ static __forceinline__ void
nvshmemi_ibgda_put_nbi_warp(uint64_t req_rptr, uint64_t req_lptr, size_t bytes, int dst_pe,
                            int qp_id, int lane_id, int message_idx) {
    uint64_t base_wqe_idx = 0, base_slot_idx = 0, base_msn_idx = 0, base_psn = 0;
    uint64_t my_slot_idx, my_msn_idx, my_psn;
    int num_wqes_per_cmd = 1, num_slots_per_wqe = 3;
    // Get lkey and rkey, store them into lanes
    uint32_t num_wqes = 0;
    __be32 my_lkey = 0;
    uint64_t my_laddr = 0;
    __be32 my_rkey = 0;
    uint64_t my_raddr = 0;
    uint64_t my_chunk_size = 0;

    // Decide how many messages (theoretically 3 for maximum)
    auto remaining_bytes = bytes;
    while (remaining_bytes > 0) {
        if (lane_id == num_wqes)
            my_chunk_size = min(remaining_bytes, ibgda_get_lkey_and_rkey(my_laddr = req_lptr, &my_lkey, req_rptr, dst_pe, &my_raddr, &my_rkey));

        // Move one more message
        auto chunk_size = __shfl_sync(0xffffffff, my_chunk_size, static_cast<int>(num_wqes));
        remaining_bytes -= chunk_size;
        req_lptr += chunk_size;
        req_rptr += chunk_size;
        ++ num_wqes;
    }
    EP_DEVICE_ASSERT(num_wqes <= 32);

    // Process WQE
    auto qp = ibgda_get_rc(dst_pe, qp_id);
    int total_msn = num_wqes;
    int ppw = bnxt_re_get_pkts_per_wqe(qp, my_chunk_size);
    int total_pkts = total_msn * ppw;

    if (lane_id == 0)
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes,
                                           num_slots_per_wqe, total_msn, total_pkts,
                                           &base_slot_idx, &base_msn_idx, &base_psn);
    base_wqe_idx = __shfl_sync(0xffffffff, base_wqe_idx, 0);

    my_slot_idx = base_slot_idx + (lane_id * num_wqes_per_cmd * num_slots_per_wqe);
    my_msn_idx = base_msn_idx + (lane_id * num_wqes_per_cmd);
    my_psn = base_psn + (lane_id * num_wqes_per_cmd * ppw);

    if (lane_id < num_wqes)
        ibgda_write_rdma_write_wqe(qp, my_laddr, my_lkey, my_raddr,
                        my_rkey, my_chunk_size, my_slot_idx,
                        my_msn_idx, my_psn, ppw);

    __syncwarp();

    // Submit
    if (lane_id == 0)
        ibgda_submit_requests<kAlwaysDoPostSend>(qp, base_wqe_idx, num_wqes,
                        base_slot_idx, num_wqes * num_slots_per_wqe, message_idx);
    __syncwarp();
}

__device__ static __forceinline__ void ibgda_write_amo_add_wqe(
        nvshmemi_ibgda_device_qp_t *qp, const int &value,
        uint64_t laddr, __be32 lkey, uint64_t raddr, __be32 rkey,
        uint16_t wqe_idx, void** out_wqes) {
    EP_DEVICE_ASSERT(true);
}

__device__ __forceinline__ void nvshmemi_ibgda_amo_nonfetch_add(void *rptr, const int& value, int pe, int qp_id, bool is_local_copy = false) {
    EP_DEVICE_ASSERT(true);
}

__device__ __forceinline__ uint64_t nvshmemi_get_p2p_ptr(const uint64_t& ptr, const int& rank, const int& dst_rank) {
    // Local rank, no need for mapping
    if (rank == dst_rank)
        return ptr;
    auto peer_base = __ldg(reinterpret_cast<uint64_t*>(nvshmemi_device_state_d.peer_heap_base_p2p) + dst_rank);

    // RDMA connected
    if (peer_base == 0)
        return 0;

    // NVLink P2P is enabled
    return peer_base + (ptr - reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base));
}

} // namespace deep_ep
