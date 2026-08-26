use cuda_device::thread;
use tuisko_model::Arch;

const VECTOR_BYTES: usize = size_of::<u128>();

#[inline(always)]
pub(crate) unsafe fn gdn_state_snapshot<A: Arch>(
    source_row: *const u32,
    history: *const u128,
    state: *const u128,
    scratch_history: *mut u128,
    scratch_state: *mut u128,
) {
    let history_bytes = A::GDN_QKV_ROWS * (A::LINEAR_CONV_KERNEL_DIM - 1) * size_of::<u16>();
    let state_bytes =
        A::GDN_CONTROL_ROWS * A::LINEAR_HEAD_DIM * A::LINEAR_HEAD_DIM * size_of::<f32>();
    let history_vectors = history_bytes / VECTOR_BYTES;
    let state_vectors = state_bytes / VECTOR_BYTES;
    let vector = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
    let row = unsafe { *source_row as usize };

    if vector < history_vectors {
        unsafe {
            *scratch_history.add(vector) = *history.add(row * history_vectors + vector);
        }
    } else if vector < history_vectors + state_vectors {
        let state_vector = vector - history_vectors;
        unsafe {
            *scratch_state.add(state_vector) = *state.add(row * state_vectors + state_vector);
        }
    }
}
