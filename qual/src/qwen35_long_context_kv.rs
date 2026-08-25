//! Device qualification for Qwen3.5 shared BF16 KV ownership.

#[cfg(test)]
mod tests {
    use crate::device_benchmark;
    use tuisko_engine::{
        PagedKvSlotState, QWEN35_LONG_CONTEXT_PHYSICAL_PAGES, QWEN35_MAX_CONTEXT_TOKENS,
        Qwen35LongContextKvLayout, Qwen35LongContextKvProgram,
    };
    use tuisko_gpu::{CudaContext, GpuError, device_memory_info};

    const QUALIFICATION_PAGES: usize = 9;

    #[test]
    fn qwen35_long_context_kv_suite_byte_accounting_is_exact() {
        let layout = Qwen35LongContextKvLayout::build().unwrap();

        assert_eq!(QWEN35_MAX_CONTEXT_TOKENS, 262_144);
        assert_eq!(QWEN35_LONG_CONTEXT_PHYSICAL_PAGES, 4_096);
        assert_eq!(layout.block_table_bytes(), 131_072);
        assert_eq!(layout.cache_bytes(), 8_589_934_592);
        assert_eq!(layout.arena_bytes(), 8_590_065_664);
    }

    #[test]
    #[ignore = "requires an exclusive RTX 5090"]
    fn qwen35_long_context_kv_suite_lifecycle_is_address_stable()
    -> Result<(), Box<dyn std::error::Error>> {
        let _preflight = device_benchmark::preflight()?;
        let context = CudaContext::new(0).map_err(GpuError::from)?;
        if context.compute_capability().map_err(GpuError::from)? != (12, 0) {
            return Err(
                "Qwen3.5 long-context KV qualification requires compute capability 12.0".into(),
            );
        }
        let stream = context.new_stream().map_err(GpuError::from)?;
        let mut program =
            Qwen35LongContextKvProgram::qualification_for_pages(&context, QUALIFICATION_PAGES)?;
        let addresses = program.qualification_addresses();

        program.activate_slot(0)?;
        program.reserve_slot_tokens(&stream, 0, 130)?;
        program.activate_slot(3)?;
        program.reserve_slot_tokens(&stream, 3, 65)?;
        stream.synchronize().map_err(GpuError::from)?;
        let tables = program.qualification_block_tables(&stream)?;
        assert_eq!(&tables[..3], &[0, 1, 2]);
        assert_eq!(&tables[3 * 4_096..3 * 4_096 + 2], &[3, 4]);
        assert_eq!(program.route(0, 129)?.physical_page(), 2);
        assert_eq!(program.route(3, 64)?.physical_page(), 4);

        let before = device_memory_info(&context)?;
        program.truncate_slot_tokens(&stream, 0, 64)?;
        program.retain_slot(0)?;
        program.activate_slot(0)?;
        program.reserve_slot_tokens(&stream, 0, 193)?;
        program.recycle_slot(&stream, 3)?;
        stream.synchronize().map_err(GpuError::from)?;
        let after = device_memory_info(&context)?;
        assert_eq!(before, after, "KV lifecycle allocated after warmup");
        assert_eq!(program.qualification_addresses(), addresses);
        assert_eq!(program.slot_token_count(0)?, 193);
        assert_eq!(program.slot_state(3)?, PagedKvSlotState::Vacant);

        program.reset(&stream)?;
        stream.synchronize().map_err(GpuError::from)?;
        assert!((0..8).all(|slot| program.slot_state(slot).unwrap() == PagedKvSlotState::Vacant));
        assert!(
            program
                .qualification_block_tables(&stream)?
                .iter()
                .all(|&page| page == u32::MAX)
        );
        device_benchmark::require_current_process_exclusive()?;
        println!(
            "Qwen3.5 long-context KV passed: {} qualification pages, {} device bytes, {} host bytes",
            program.physical_pages(),
            program.arena_bytes(),
            program.host_allocation_bytes()
        );

        Ok(())
    }
}
