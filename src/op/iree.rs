
fn example() {
  // create a runtime instance
    let instance = IreeRuntimeInstance::try_from_options(
        &IreeRuntimeInstanceOptionsBuilder::default()
            .use_all_available_drivers()
            .build(),
        &IreeAllocator::system_allocator(),
    )?;

    // create a device
    let device = instance.try_create_default_device("local-task")?;

    // get host allocator
    let allocator = instance.host_allocator();

    // create a session
    let session = IreeRuntimeSession::create_with_device(
        &instance,
        &IreeRuntimeSessionOptionsBuilder::default().build(),
        &device,
        &allocator,
    )?;

    // load resnet18 vmfb to session
    session.append_bytecode_module_from_memory(RESNET18_VMFB.as_slice(), &allocator)?;

    // // get the entry function
    let mut call = session.get_call_by_name("module.forward")?;

    // load input image
    let j: Image = serde_json::from_slice(&TEST_IMAGE).unwrap();

    // get device allocator
    let device_allocator = session.device_allocator();

    // convert image to const byte span
    let bytespan = IreeConstByteSpan::from_slice(&j.data);
    let image_shape = j.shape;
    let buffer_params = IreeHalBufferViewParamsBuilder::default()
        .type_(iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL.0)
        .access(0)
        .usage(iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_DEFAULT.0)
        .build();

    // create hal buffer view
    let input = IreeHalBufferView::allocate_buffer(
        &device_allocator,
        &image_shape,
        iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        iree_hal_encoding_types_t_IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        &buffer_params,
        &bytespan,
    )?;

    // push input to call
    call.inputs_push_back_buffer_view(&input)?;

    // invoke call
    call.invoke(iree_runtime_call_flags_t::default())?;

    // pop output from call
    let output = call.outputs_pop_front_buffer_view()?;

    println!("output: {}", output);

    Ok(())
}
