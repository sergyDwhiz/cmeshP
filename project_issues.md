# Project Issues & Checklist

## Setup & Installation
- [ ] Verify that Homebrew installation works reliably on all supported macOS configurations.
- [ ] Ensure the script correctly adds Homebrew to PATH for Apple Silicon Macs.
- [ ] Validate that the installation directory (/usr/local/freesurfer) is created with proper permissions.
- [ ] Confirm that the install_freesurfer.sh script has been pasted into the workspace.

## Download & Extraction
- [ ] Check that the FreeSurfer download URL is current and reliable.
- [ ] Verify the downloaded file integrity:
  - [x] Ensure the file exists after download.
  - [x] Confirm the file size meets expectations (>100MB) to avoid error pages.
- [ ] Test archive extraction and verify that the extracted content is valid.
- [ ] Ensure the correct archive file is present before extraction during manual installation (e.g., verify filename/version matches, such as 7.4.1 vs 7.1.1).

## Environment Variables & License
- [ ] Ensure environment variables (FREESURFER_HOME and SetUpFreeSurfer.sh sourcing) are correctly appended to ~/.zshrc.
- [ ] Confirm that duplicate entries are not added on subsequent runs.
- [ ] Provide clear user instructions for obtaining and placing the required license file.

## Error Handling & Future Improvements
- [ ] Improve error messages and exit conditions for better user feedback.
- [ ] Validate that the script fails gracefully in case of network issues or extraction errors.
- [ ] Consider cross-platform support for Linux/macOS if needed.
- [ ] Refine download method consistency (wget vs curl) based on system availability.
- [ ] Evaluate additional dependencies or prerequisites that may affect the installation.
