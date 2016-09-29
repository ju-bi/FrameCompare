# FrameCompare

An OpenCV tool used to compare a frame to a referent frame and displaying differences.

## Usage:

```dos 
  frcmp <path_to_reference_frame> <path_to_frame_to_compare> 
    [--quiet <mse_threshold> <ssi_threshold> [<path_to save_diff_frame>]]
```
  
## Exit code (quiet mode only):
  
* 2 if error in command-lines arguments (print usage)
* 1 if frames are not similar, according to specified thresholds (quiet mode only)
* 0 otherwise
 
## References (MSE and SSI using OpenCV):
 
* http://stackoverflow.com/questions/4196453/simple-and-fast-method-to-compare-images-for-similarity
* http://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
* http://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html#videoinputpsnrmssim
