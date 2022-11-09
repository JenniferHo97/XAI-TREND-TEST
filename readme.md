# Trend-based Faithfulness Test

This is the code for our paper which is submitted to ISSTA 2023.

We use `python3.7` to run this code with some necessary packages:

```text
captum==0.4.0
pytorch==1.9.0+cu111
torchtext==0.10.0
torchvision==0.10.0+cu111
gensim==4.1.2
numpy==1.20.3

```

This version is only used for paper review. In the future, we will open source our codes in the form of docker to make it easier for test.

## Downstream Application: Model Debugging

Model debugging is one of the ways to uncover spurious correlations learned by the model and help the users improve their models. For example, consider a classification task where all the airplanes in the dataset always appear together with the background (i.e., the blue sky). The model might then correlate the background features of the blue sky with the airplane category during training. This spurious correlation indicates that the model learns different category knowledge from what users envision, making the model vulnerable and insecure. If the users can detect the spurious correlation, they could enlarge the data space or deploy a stable deep learning module during training. In previous studies, explanation techniques are often used to diagnose spurious correlation. However, different explanation methods perform differently. For example, in the following figure, IG considers that the model focuses on both object and background while SG-IG-SQ marks the blue sky background as the important feature. We could not make sure which explanation is more conformed to the model.

[!vis_debug](images/vis_debug.png)

In this section, we will verify the effectiveness of our trend tests on guiding the user to choose an explanation method for model debugging and then examine the performance of explanation methods on detecting spurious correlations. Based on Adebayo \etal~\cite{modeldebug}, we construct a model with known spurious correlation and use the trend-based tests on the model to observe the faithfulness of each explanation method. Then, we analyze the model with explanation methods to see whether the explanation results focus on the spurious correlated features. Next, we could verify whether the results of the trend-based tests are consistent with the results of the debugging test. We extract the object of cats and planes from MSCOCO2017~\cite{MSCOCO}, and then replace the backgrounds with the bedroom and the coast from MiniPlaces~\cite{miniplaces}, respectively. We synthesize eight types of data as shown in Figure~\ref{fig:coco_data_vis}. $D_{airplane-coast}$ means the object is an airplane, and the context is the coast. Each of them includes $1000$ pictures. We use the first two ($D_{airplane-coast}$ and $D_{cat-bedroom}$) to train a ResNet18 model. We split the training data into a training set and a validate set at a ratio of 8:2. The rest are used for testing. The accuracy of the model is shown in Table~\ref{eva:debug_model}.
