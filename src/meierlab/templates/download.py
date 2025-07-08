"""
XNAT Download Template for meierlab data processing workflows.

This module provides a template for downloading data from XNAT servers
with support for DICOM to NIfTI conversion and BIDS organization.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from zipfile import ZipFile

from .base import BaseDownloadTemplate
from ..cirxnat import Cirxnat


class XNATDownloadTemplate(BaseDownloadTemplate):
    """
    XNAT Download Template for downloading data from XNAT servers.
    
    This template provides functionality for downloading data from XNAT servers
    with support for DICOM to NIfTI conversion and BIDS organization.
    Users can use this template as-is or inherit from it to customize behavior.
    
    Examples
    --------
    Basic usage:
    
    >>> from meierlab.templates import XNATDownloadTemplate
    >>> template = XNATDownloadTemplate()
    >>> template.run()
    
    Custom configuration:
    
    >>> config = {
    ...     'address': 'https://my-xnat-server.com',
    ...     'project': 'MY_PROJECT',
    ...     'working_directory': '/path/to/data'
    ... }
    >>> template = XNATDownloadTemplate(config)
    >>> template.run()
    
    Command line usage:
    
    >>> python -m meierlab.templates.download --help
    """
    
    __version__ = "1.0.0"
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for XNAT download template.
        
        Returns
        -------
        dict
            Default configuration dictionary
        """
        return {
            'address': 'https://cirxnat2.rcc.mcw.edu',
            'project': 'CSI_MRI_MCW',
            'username': None,
            'password': None,
            'working_directory': '/scratch/g/mmccrea/CSI/testing',
            'subject': '',
            'experiment': '',
            'list_file': '',
            'dcm2nii': False,
            'bids': False,
            'scan': '',
            'scan_list': [],
            'dry_run': False
        }
    
    def _validate_config(self) -> None:
        """
        Validate the configuration.
        
        Raises
        ------
        ValueError
            If configuration is invalid
        """
        required_fields = ['address', 'project', 'working_directory']
        for field in required_fields:
            if not self.config.get(field):
                raise ValueError(f"Required configuration field '{field}' is missing or empty")
        
        if not self.config.get('username') or not self.config.get('password'):
            raise ValueError("Username and password are required for XNAT authentication")
    
    def _add_arguments(self, parser):
        """
        Add XNAT-specific command line arguments.
        
        Parameters
        ----------
        parser : argparse.ArgumentParser
            Argument parser to add arguments to
        """
        parser.add_argument(
            '-a', '--address',
            help='XNAT instance address',
            default=self.config['address']
        )
        parser.add_argument(
            '-p', '--project',
            help='XNAT project name',
            default=self.config['project']
        )
        parser.add_argument(
            '-un', '--username',
            help='XNAT username',
            required=True
        )
        parser.add_argument(
            '-up', '--password',
            help='XNAT password',
            required=True
        )
        parser.add_argument(
            '-wd', '--working-directory',
            help='Processing directory',
            default=self.config['working_directory']
        )
        parser.add_argument(
            '-s', '--subject',
            help='Download specific subject. Use "all" for all subjects',
            default=self.config['subject']
        )
        parser.add_argument(
            '-e', '--experiment',
            help='Download specific experiment. Use "all" for all experiments',
            default=self.config['experiment']
        )
        parser.add_argument(
            '-l', '--list',
            help='List of subjects/experiments to download',
            default=self.config['list_file']
        )
        parser.add_argument(
            '-d', '--dcm2nii',
            help='Convert DICOM to NIfTI',
            action='store_true'
        )
        parser.add_argument(
            '-b', '--bids',
            help='Convert to BIDS format',
            action='store_true'
        )
        parser.add_argument(
            '-sc', '--scan',
            help='Download specific scan. Use "all" for all scans',
            default=self.config['scan']
        )
        parser.add_argument(
            '-st', '--scan-list',
            nargs="+",
            help='Download specific scan list',
            default=self.config['scan_list']
        )
    
    def run(self) -> None:
        """
        Run the XNAT download template.
        """
        # Parse command line arguments
        args = self.parse_arguments()
        
        # Update config with command line arguments
        self.config.update({
            'address': args.address,
            'project': args.project,
            'username': args.username,
            'password': args.password,
            'working_directory': args.working_directory,
            'subject': args.subject,
            'experiment': args.experiment,
            'list_file': args.list,
            'dcm2nii': args.dcm2nii,
            'bids': args.bids,
            'scan': args.scan,
            'scan_list': args.scan_list,
            'dry_run': args.dry_run
        })
        
        # Initialize XNAT connection
        self.server = Cirxnat(
            self.config['address'],
            self.config['project'],
            self.config['username'],
            self.config['password']
        )
        
        # Get all experiments
        all_experiments = self.server.print_all_experiments()
        self.logger.info(f"Found {len(all_experiments)} experiments")
        
        # Get download dictionary
        download_dict = self._get_download_dict(all_experiments)
        
        if self.config['dry_run']:
            self.logger.info("DRY RUN: Would download the following:")
            for bids_label, experiment in download_dict.items():
                self.logger.info(f"  {bids_label}: {experiment['scan_list']}")
            return
        
        # Setup and execute downloads
        self._setup(download_dict)
        self.logger.info("Download completed successfully")
    
    def _get_download_dict(self, all_experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get the download dictionary based on configuration.
        
        Parameters
        ----------
        all_experiments : list
            List of experiments from XNAT
            
        Returns
        -------
        dict
            Dictionary of download information
        """
        download_dict = {}
        
        for experiment in all_experiments:
            subj_label = experiment['subject_label']
            exp_label = experiment['experiment_label']
            
            # Create BIDS label
            if self.config['bids']:
                bids_exp = exp_label.replace('_', '').replace('-', '')
                bids_label = f'sub-{bids_exp}'
            else:
                bids_label = exp_label
            
            # Check if experiment should be included
            should_include = False
            
            if self.config['list_file'] and os.path.exists(self.config['list_file']):
                self.logger.info("Reading from list file...")
                with open(self.config['list_file'], 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip('\n') == subj_label or line.strip('\n') == exp_label:
                            should_include = True
                            break
            elif (self.config['subject'] != 'all' and 
                  self.config['experiment'] != 'all'):
                if (self.config['subject'] == subj_label and 
                    self.config['experiment'] == exp_label):
                    should_include = True
            else:
                should_include = True
            
            if should_include:
                download_dict[bids_label] = experiment
                self.logger.info(f"Added {bids_label} to download list")
            
            # Get scan list
            scan_dict = self.server.get_scans_dictionary(subj_label, exp_label)
            scan_list = []
            
            if self.config['scan'] != 'all' and not self.config['scan_list']:
                if self.config['scan']:
                    scan_list.append(self.config['scan'])
            elif self.config['scan'] != 'all' and self.config['scan_list']:
                scan_list = self.config['scan_list']
            elif self.config['scan'] == 'all':
                scan_list = list(scan_dict.values())
            else:
                raise ValueError("No scans specified for download")
            
            if bids_label in download_dict:
                download_dict[bids_label]['scan_list'] = scan_list
        
        return download_dict
    
    def _setup(self, download_dict: Dict[str, Any]) -> None:
        """
        Setup and execute downloads.
        
        Parameters
        ----------
        download_dict : dict
            Dictionary of download information
        """
        for bids_label, experiment in download_dict.items():
            subj_label = experiment['subject_label']
            exp_label = experiment['experiment_label']
            scan_list = experiment['scan_list']
            
            # Create temporary directory
            tmp_dir = os.path.join(
                self.config['working_directory'],
                'sourcedata',
                bids_label
            )
            self.create_folder(tmp_dir)
            
            # Download scan descriptions
            self.server.zip_scan_descriptions_to_file(
                subj_label, exp_label, scan_list, f'{bids_label}.zip'
            )
            
            # Extract downloaded files
            try:
                with ZipFile(f"{self.config['working_directory']}/{bids_label}.zip", "r") as ff:
                    ff.extractall(tmp_dir)
                os.remove(f'{self.config["working_directory"]}/{bids_label}.zip')
            except Exception as e:
                self.logger.error(f"Failed to extract {bids_label}.zip: {e}")
                continue
            
            # Convert DICOM to NIfTI if requested
            if self.config['dcm2nii']:
                scan_dirs = list(Path(tmp_dir).glob('**/scans/*'))
                for scan_dir in scan_dirs:
                    if self.config['bids']:
                        self._scan_to_bids(scan_dir, bids_label, tmp_dir)
                    else:
                        self._dcm2nii(scan_dir, tmp_dir, bids_label)
            
            # Update permissions
            self.update_permissions(tmp_dir)
    
    def _scan_to_bids(self, scan_dir: Path, bids_label: str, tmp_dir: str) -> None:
        """
        Convert a scan to BIDS format.
        
        Parameters
        ----------
        scan_dir : Path
            Path to the scan directory
        bids_label : str
            BIDS label for the scan
        tmp_dir : str
            Path to the temporary directory
        """
        bids_name = ''
        s_name = str(scan_dir.name).lower()
        self.logger.info(f"Processing scan: {s_name}")
        
        if "mprage" in s_name or "bravo" in s_name:
            bids_name = 'T1w'
        elif "dwi" in s_name or "dti" in s_name or "diffusion" in s_name:
            bids_name = 'dwi'
            if "pa" in s_name:
                bids_name = 'dir-PA_epi'
            elif "ap" in s_name:
                bids_name = 'dir-AP_epi'
        elif "t2" in s_name:
            bids_name = 'T2w'
        elif "flair" in s_name:
            bids_name = 'FLAIR'
        elif "fmri" in s_name:
            bids_name = 'task'
        elif "asl" in s_name or 'blood' in s_name:
            bids_name = 'asl'
        else:
            bids_name = 'unknown'
        
        self._dcm2nii(scan_dir, tmp_dir, f'{bids_label}_{bids_name}')
    
    def _dcm2nii(self, dcm_dir: Path, nii_dir: str, bids_label: str) -> None:
        """
        Convert DICOM files to NIfTI files.
        
        Parameters
        ----------
        dcm_dir : Path
            Path to the DICOM directory
        nii_dir : str
            Path to the NIfTI directory
        bids_label : str
            BIDS label for the scan
        """
        dcm_cmd = f'dcm2niix -o {nii_dir} -f {bids_label} -z i -b y -m y {dcm_dir}'
        try:
            result = subprocess.run(
                dcm_cmd, check=True, capture_output=True, shell=True, text=True
            )
            self.logger.info(f"DICOM to NIfTI conversion successful: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"DICOM to NIfTI conversion failed: {e.stderr}")


def main():
    """Main entry point for the XNAT download template."""
    template = XNATDownloadTemplate()
    template.run()


if __name__ == "__main__":
    main()
