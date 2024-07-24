from pathlib import Path


class FreeSurfer:
    """
    Base class to access FreeSurfer-specific files.
    """

    def __init__(self, home_dir, subjects_dir, subject_id):
        self.home_dir = Path(home_dir)
        self.subjects_dir = Path(subjects_dir)
        self.subject_id = subject_id
        self.recon_success = self.check_recon_all()

    def get_stats(self, file_name):
        stats_file = self.subjects_dir / self.subject_id / "stats" / file_name

        return stats_file

    def check_recon_all(self):
        recon_file = self.subjects_dir / self.subject_id / "scripts/recon-all.log"

        if recon_file.exists():
            with open(recon_file) as reconall:
                line = reconall.readlines()[-1]
                if "finished without error" in line:
                    return True
                else:
                    return False
        else:
            raise Exception("Subject has no recon-all output.")
