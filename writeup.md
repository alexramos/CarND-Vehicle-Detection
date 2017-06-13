# Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test\_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/original_imgs.png
[image2]: ./output_images/features.png
[image4]: ./output_images/finding_cars.png
[image5]: ./output_images/heatmap_generation.png
[image6]: ./output_images/labels.png
[image7]: ./output_images/final_bounds.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in code cells 2-5 of the IPython notebook "vehicle\_detection\_dev.ipynb".

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `L` channel of the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` as these yielded feature images that distinguished cars from not-cars.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a classifier to predict car images in code cells 11-14 of the IPython notebook "vehicle\_detection\_dev.ipynb".  Here I used a Gaussian RBF SVM with parameters `C=10` and `gamma='auto'`.  I arrived at these hyperparameters by using sci-kit-learn's GridSearchCV function.  After fitting the model with the training set, it yield a classification accuracy of 99% on the test set.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To search for cars in an image I used the HOG sub-sampling window search method shown in lesson 35 of the project lessons.  This approach is notable in that it performs HOG feature extraction only once on the entire image and is thus much more efficient than performing HOG feature extraction on each search window individually.  I decided to search on two scales, `1.5` and `2.1`, with an overlap of `75%` as these setting provided the best sensitivity for finding cars in the test images provided.

This is implemented in code cells 15-17 of the IPython notebook "vehicle\_detection\_dev.ipynb".

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HLS 1-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

This is implemented in code cells 18-24 of the IPython notebook "vehicle\_detection\_dev.ipynb".

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took is largely based off the examples provided in this section's lessons. 
First I implemented a pipeline that extracts HOG features, spatial bins, and color histograms from a image.  I then used this feature extraction pipeline on a training dataset of car/not-car images.  With training features in-hand, I then trained SVM classifer to predict if an image contains a car.  Lastly, I performed a sliding window search on subimages to classify individuals cars in a video image.

My implementation works OK to identify cars in a video stream but results in wobbly bounding boxes that are constantly changing size.  My implementation performs poorly when only a portion of a car is in the video, and also when two cars are very close to each other, leading to a single large bounding box over both cars.

If I were going to persue this project further, I would make the pipeline more robust by using a deep learning classification algorithm that is not dependent on predefined features.




