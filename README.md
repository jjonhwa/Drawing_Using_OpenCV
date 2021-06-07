#  Line Drawing at Table Using OpenCV

This Course is one of the challenges addressed in parsing appraisal papers. In the case of evaluation report, the form of the table is difficult to parse using the existing table extraction library. The form of the table is not divided into cells, and in order to parse it correctly, a line was drawn to make the cell shape.

This Course is antecedent of the following GitHub - [Table Extraction by Kor](https://github.com/jjonhwa/Table_Extraction_Kor-benchmark). 

In addition, there is an understandable explanation for OpenCV with code, and it can be applied if the shape of the table is not what we know.

## Idea
In my case, there is a text with several supporting explanations, plus a table. 

Therefore, in order to create a cell form on a table that is different from the existing table format, the existing vertical and horizontal lines are deleted and underlined based on text to create a horizontal line.  

We obtain the coordinates of the drawn lines and draw them on the original table to form the table we commonly know.

## Course
All the processes were carried out in Colab. (CPU)  
**Note:** cv2.imshow has been changed from Colab to cv2_imshow. If you don't use it in Colab, please modify the code and use it.

The order is as follows.
**Image scale** > **Cut Image** > **Search max, min Location x** > **Remove vertical line and horizontal line** > **Dilate** > **Erode** >  **Find location**

## Examples & Demo Notebook

| Original Table | Drawing Table |
|-----|-----|
|![Original_Document](https://user-images.githubusercontent.com/53552847/120768675-60b17000-c557-11eb-8fb9-da536b45f63c.png)|![Drawing_Document](https://user-images.githubusercontent.com/53552847/120768683-61e29d00-c557-11eb-81df-60cb5c20cc0f.png)|

**Note :** In this Notebook, The focus is on appraisal reports. so if you want to apply the above method to your documents, you have to modify the code and use it.  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SHQQ3WWO90cKSwJAAbvld641S-ouRg1b#scrollTo=2qc2CPkXKh4t)

Please check [Code_Review](https://jjonhwa.github.io/2021-06-07-Code_Review/) & [Code Explanation](https://jjonhwa.github.io/2021-06-06-Code_Explanation/) for an explanation of OpenCV.
