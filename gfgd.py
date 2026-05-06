def __detect_parts(self) -> None:
    """
    This function detects the parts using the detector dataclass.

    It utilises multi-threading to speed up when multiple detectors are
    used. In theory it should only work marginally do to the bottleneck of
    using the GPU (or CPU), however it's still a minor improvement.

    """
    if self.cache and self.cache.check_for_frame(self.frame_counter):
        all_parts = self.cache.get_frame(self.frame_counter).output_data
    else:
        with ThreadPoolExecutor() as executor:
            detected_parts = list(
                executor.map(
                    lambda detector: detector.detect_image(self.file_image),
                    enabled_detectors,
                ),
            )

        all_parts = list(itertools.chain(*detected_parts))
        output = AIOutputData(model_name="nude_net", output_data=all_parts)
        if self.cache:
            self.cache.save_frame(self.frame_counter, output)

    # Sort and Label ID Based on Position
    all_parts.sort(
        key=lambda part: (part.relative_box[1], part.relative_box[0]),
    )

    for index, part in enumerate(all_parts, start=1):
        part.set_part_id(index)

    self._detected_parts = all_parts
