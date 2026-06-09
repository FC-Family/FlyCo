from sam_tk.config.model_args import sam2_args


def tracker_factory(tracker_type, args=None):
    if tracker_type != "sam2camera":
        raise ValueError(
            f"Unsupported tracker_type={tracker_type!r}. Only 'sam2camera' is kept."
        )

    from sam_tk.utils.SamTracker import SamTracker

    merged_args = dict(sam2_args)
    if args is not None:
        merged_args.update(args)
    return SamTracker(merged_args)
