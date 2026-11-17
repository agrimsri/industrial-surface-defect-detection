# Overview

Modern manufacturing relies on automated visual inspection to catch quality issues before products leave the production line.

In this competition you will build an AI inspection system that finds surface anomalies in industrial images, identifies their type, and marks exactly which pixels they cover. It is not enough to say whether an image contains a defect: your model must localize every anomaly at the pixel level, for four different anomaly categories.

Submissions are scored on how accurately the predicted masks match the true anomaly regions, and on how reliably the model avoids reporting anomalies that are not there.

# Evaluation

Submissions are evaluated using a weighted segmentation metric designed specifically for this competition.

The final score combines three components:

## 1\. Foreground Dice Score: 70%

This measures how accurately the predicted anomaly region overlaps with the ground truth region when an anomaly is actually present.

The Dice score is calculated as:

Dice \= (2 \* \|X ∩ Y\|) / (\|X\| \+ \|Y\|)

* X is the predicted set of pixels.  
* Y is the ground truth set of pixels.

A higher Dice score means better pixel-level overlap. This is the most important component of the competition score.

## 2\. False Positive Score: 20%

A prediction should not report an anomaly where none exists.

This component penalizes unnecessary anomaly predictions on image and category pairs whose ground truth contains no anomaly.

## 3\. Empty Prediction Accuracy: 10%

The model should also correctly recognize when a particular anomaly category is absent. Correctly predicting an empty mask contributes positively to this component.

## Final Score

The score ranges from 0 to 1, with higher scores indicating better performance.

This design rewards models that can both accurately localize real anomalies and avoid unnecessary false detections.

# Submission Format

Your submission must contain the following columns:

```
ID,image_id,defect_type,mask_rle
```

There should be one prediction for every combination of **image\_id × defect\_type**.

For example:

```
ID,image_id,defect_type,mask_rle
1,WAFNKCRR7Q,defect_a,""
2,WAFNKCRR7Q,defect_b,3201 12 3329 12 3457 12
3,WAFNKCRR7Q,defect_c,""
4,WAFNKCRR7Q,defect_d,""
```

An empty **mask\_rle** indicates that the model predicts no pixels belonging to that anomaly category.

Use the provided **sample\_submission.csv** as the definitive reference for the submission structure and column order.

# Dataset Description

The competition provides a collection of industrial inspection images together with pixel level annotations for the training set.

Each training annotation contains:

| Column | Description |
| :---- | :---- |
| image\_id | Anonymous identifier of the image |
| defect\_type | One of the four anomaly categories |
| mask\_rle | Run length encoded segmentation mask |

Each image can contain:

* no anomaly of a particular category,  
* one anomaly category,  
* multiple anomaly categories,  
* multiple disconnected regions belonging to the same category.

The test set contains images without publicly available annotations. Your model must generate predictions for these unseen images.

## Anomaly Variability

Surface anomalies can vary significantly in:

* size,  
* shape,  
* position,  
* orientation,  
* texture,  
* contrast,  
* number of occurrences.

Some anomalies may occupy only a small portion of an image, while others can cover much larger regions.

## Run Length Encoding

To keep submission files compact, segmentation masks are represented using **Run Length Encoding (RLE)**. Instead of submitting every pixel individually, consecutive foreground pixels are represented using *start length* pairs.

For example, **10 5** represents five consecutive foreground pixels beginning at position 10\. Multiple regions can be represented by multiple pairs, such as **10 5 30 4 100 8**.

The masks use column-major ordering.

Participants do not need to manually construct the encoded masks if their segmentation pipeline already provides binary masks. However, the final predictions must be converted into the required RLE format before submission.

## Dataset Metadata

| Files | 14,395 files |
| :---- | :---- |
| **Size** | 390.97 MB |
| **File Types** | JPG, CSV |

