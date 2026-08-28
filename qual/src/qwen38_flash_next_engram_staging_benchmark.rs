//! Accounting and source-backed timing for Qwen3.8-Flash-Next engram staging.

#[cfg(test)]
mod tests {
    use crate::device_benchmark;
    use std::hint::black_box;
    use std::path::Path;
    use std::time::Instant;
    use tuisko_engine::{
        LayerMemoryLayout, QWEN38_FLASH_NEXT_ENGRAM_WIDTHS, Qwen38FlashNextEngramStager,
        Qwen38FlashNextEngramStagerLayout, StreamingResidencyAccounting,
        gather_qwen38_flash_next_engram_window,
    };
    use tuisko_gpu::{CudaContext, GpuError, device_memory_info};
    use tuisko_model::{
        CheckpointSnapshot, QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN,
        QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN, Qwen38FlashNext, Qwen38FlashNextEngramBindings,
        Qwen38FlashNextEngramCarry,
    };

    type Suite = Result<(), Box<dyn std::error::Error>>;

    const SNAPSHOT_ENVIRONMENT: &str = "TUISKO_QWEN38_FLASH_NEXT_SNAPSHOT";
    const STAGED_DIGEST: u64 = 0xda0f_3e47_48e0_ff76;

    fn tokens(len: usize) -> Vec<u32> {
        (0..len)
            .map(|index| {
                if index % 97 == 96 {
                    QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN
                } else {
                    ((index * 7_919 + 17) % 248_319) as u32
                }
            })
            .collect()
    }

    fn digest_tokens() -> Vec<u32> {
        let mut state = 0x5f3a_9c21u64;
        let mut window = (0..128)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) % 248_319) as u32
            })
            .collect::<Vec<_>>();
        for position in [0, 1, 63, 127] {
            window[position] = QWEN38_FLASH_NEXT_ENGRAM_EOS_TOKEN;
        }
        window
    }

    fn digest(staged: &[u8]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in staged.iter().copied().chain(staged.len().to_le_bytes()) {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    #[test]
    fn qwen38_flash_next_engram_staging_suite_benchmark_accounting_is_exact() {
        let layout = Qwen38FlashNextEngramStagerLayout::build().unwrap();

        assert_eq!(QWEN38_FLASH_NEXT_ENGRAM_WIDTHS, [1, 32, 64, 128, 1_024]);
        assert_eq!(layout.max_tokens(), 1_024);
        assert_eq!(layout.token_bytes(), 2_560);
        assert_eq!(layout.round_rows(1_024).unwrap(), 16_384);
        assert_eq!(layout.round_bytes(1_024).unwrap(), 2_621_440);
        assert_eq!(layout.row_scratch_bytes(), 131_072);
        assert_eq!(layout.table_bytes(), 51_200_245_760);
        assert_eq!(layout.arena_bytes(), 2_621_440);
        assert_eq!(layout.workspace_bytes(), 2_621_440);
        assert_eq!(layout.resident_weight_bytes(), 0);
        assert_eq!(layout.cache_bytes(), 0);
        assert_eq!(layout.device_resident_bytes(), 2_621_440);
        assert_eq!(layout.host_pinned_bytes(), 2_621_440);
        assert_eq!(layout.host_mapped_bytes(), 51_200_245_760);
    }

    #[test]
    #[ignore = "requires an exclusive RTX 5090 and the pinned Qwen3.8 snapshot"]
    fn qwen38_flash_next_engram_staging_suite_benchmark_times_the_source_backed_owner() -> Suite {
        let Some(root) = std::env::var_os(SNAPSHOT_ENVIRONMENT) else {
            println!(
                "SKIPPED Qwen3.8 Engram source benchmark: set {SNAPSHOT_ENVIRONMENT} to the pinned snapshot"
            );
            return Ok(());
        };

        let _preflight = device_benchmark::preflight()?;
        let snapshot = CheckpointSnapshot::<Qwen38FlashNext>::open(Path::new(&root))?;
        let engram = Qwen38FlashNextEngramBindings::bind(&snapshot, Qwen38FlashNext::PLE_LAYER)?
            .materialize()?;
        let table = engram.table()?;

        let digest_window = digest_tokens();
        let mut digest_rows =
            vec![0; digest_window.len() * QWEN38_FLASH_NEXT_ENGRAM_ROWS_PER_TOKEN];
        let mut digest_staged = vec![0; digest_window.len() * table.token_bytes()];
        let mut digest_carry = Qwen38FlashNextEngramCarry::start();
        gather_qwen38_flash_next_engram_window(
            table,
            &mut digest_carry,
            &digest_window,
            &mut digest_rows,
            &mut digest_staged,
        )?;
        for (&row, gathered) in digest_rows
            .iter()
            .zip(digest_staged.chunks_exact(table.head_dim()))
        {
            assert_eq!(table.row_codes(row)?, gathered);
        }
        assert_eq!(digest(&digest_staged), STAGED_DIGEST);

        let context = CudaContext::new(0).map_err(GpuError::from)?;
        if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
            return Err("engram staging benchmark requires compute capability 12.0".into());
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let mut stager = Qwen38FlashNextEngramStager::new(&context)?;
        let window = tokens(1_024);
        let mut carry = Qwen38FlashNextEngramCarry::start();

        for _ in 0..4 {
            stager.stage(&stream, table, &mut carry, black_box(&window))?;
        }
        stream.synchronize().map_err(GpuError::from)?;

        let before = device_memory_info(&context)?;
        let started = Instant::now();
        const REPETITIONS: usize = 16;
        for _ in 0..REPETITIONS {
            stager.stage(&stream, table, &mut carry, black_box(&window))?;
        }
        stream.synchronize().map_err(GpuError::from)?;
        let elapsed = started.elapsed();
        let after = device_memory_info(&context)?;

        assert_eq!(
            before, after,
            "source-backed staging allocated after warmup"
        );
        assert_eq!(stager.staged_tokens(), 1_024);
        assert_eq!(stager.staged_window()?.len(), 2_621_440);
        device_benchmark::require_current_process_exclusive()?;
        println!(
            "Qwen3.8 Engram staging: {REPETITIONS} source-backed T=1024 rounds in {:.3} s, {:.3} ms/round",
            elapsed.as_secs_f64(),
            elapsed.as_secs_f64() * 1_000.0 / REPETITIONS as f64
        );
        println!("diagnostic only; this report cannot bless a performance baseline");

        Ok(())
    }
}
