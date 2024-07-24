from meierlab.freesurfer import FreeSurfer


def test_get_stats(
    fake_freesurfer_home, fake_subjects_dir, fake_recon_all, example_subject_id
):
    stats_dir = fake_subjects_dir / f"{example_subject_id}/stats"
    stats_dir.mkdir(parents=True)
    stats_file = stats_dir / "aseg.stats"
    stats_file.write_text("stats file")

    fs_dir = FreeSurfer(fake_freesurfer_home, fake_subjects_dir, example_subject_id)

    assert fs_dir.get_stats("aseg.stats").exists()


def test_check_recon_all_success(
    fake_freesurfer_home, fake_subjects_dir, example_subject_id
):
    scripts_dir = fake_subjects_dir / f"{example_subject_id}/scripts"
    scripts_dir.mkdir(parents=True)
    recon_file = scripts_dir / "recon-all.log"
    recon_file.write_text("finished without error")

    fs_dir = FreeSurfer(fake_freesurfer_home, fake_subjects_dir, example_subject_id)

    assert fs_dir.check_recon_all()
    assert fs_dir.recon_success


def test_check_recon_all_failure(
    fake_freesurfer_home, fake_subjects_dir, example_subject_id
):
    scripts_dir = fake_subjects_dir / f"{example_subject_id}/scripts"
    scripts_dir.mkdir(parents=True)
    recon_file = scripts_dir / "recon-all.log"
    recon_file.write_text("ERROR")

    fs_dir = FreeSurfer(fake_freesurfer_home, fake_subjects_dir, example_subject_id)

    assert not fs_dir.check_recon_all()
    assert not fs_dir.recon_success
