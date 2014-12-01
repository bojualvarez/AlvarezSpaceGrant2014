%This Script will unify a few others in order to read in a large set of
%image data from a NASA HiRise image that can be classified according to
%image intensity using a similar SVM algorithm to the one used for
%Iris_SVM. The features used for SVM will be the grayscale values of each
%pixel in the array of the HiRise image

%First the data must be read into the program. This will be accomplished
%using Leon's code for reading in and pre-processing the images.

%Read in the image and crop it according to user specification so that only
%the area of interest is analyzed. %Output a figure called general_figure
%under this file that will allow the user to manipulate the image
%appropriately

%The first part of this script is used to generate publishable images in matlab The main
%output is a matlab fig file called general_figure with the desired image
%and an image without the scale located in the file Data\Resuts\named
%image.png Due to matlab's poor manipulation of images, it is easy to use the
%fig utility in mtlab to save the figure file>>Save>> and select the
%desired format.%This script reads the given JP2 images and crops them. The
%user selects the upper left and lower right corners in the original image.

clear
close all

[image, pathimage]=uigetfile('.jp2','Select the image you would like to process');
reduction_scale = input('Please specify the reduction scale (Resolution of the image will be reduced by 2^x where x is the number you specify): ');
%Now we load the image before the slump
img = imread(image, 'ReductionLevel', reduction_scale);
%Next we choose the points that will serve as our upper left corner and
%lower right corner in the cropped image (i.e. that part of the image with
%rootless cones)
points = readPoints_v2(img, 2);
%readPoints_v2 comes out with an 2xn matrix with n being the number of points
%specified. Thus, the top left corner is in points(:,1) and the bottom
%right corner is in points(:,2)
cropped_image = img(points(2,1):points(2,2),points(1,1):points(1,2));

%First we save the files for analysis, these files have no scale image
folder = strcat(pathimage, 'Data\Results\');
filename = strcat('cropped', image, '.png');
filename = [folder filename];
imwrite(cropped_image,filename)

%Now read the grayscale pixel value for each pixel in the new figure and
%define this value as Y
Cones_Params=imread(filename,'png');

%Now we need to train the SVM by choosing points that we think are rootless
%cones, drawing squares around them and then finding the intensity values
%at these images. We will collect a data set of as many cones are in the
%region the user selected

[cones,n]=readPoints_v2(Cones_Params);

Cones_Square=zeros(size(Cones_Params));
for i=1:n
    Cones_Square(cones(2,i)-10:cones(2,i)+10,cones(1,i)-10:cones(1,i)+10)=Cones_Params(cones(2,i)-10:cones(2,i)+10,cones(1,i)-10:cones(1,i)+10);
end
Cones_Labels=logical(Cones_Square);

Flattened_Cones_Params=double(reshape(Cones_Params,[numel(Cones_Params),1]));
Flattened_Cones_Labels=double(reshape(Cones_Labels,[numel(Cones_Labels),1]));

for i=1:3
    if i==1
        Kernel_Func='linear';
    elseif i==2
        Kernel_Func='rbf';
    elseif i==3
        Kernel_Func='polynomial';
    end
    
    %Matlab utilizes the fitcsvm function to classify data. The entire data
    %set, Flattened_Cones_Params, is used as the training set. Flattened_Cones_Labels is used as the labels. Then the
    %following options are used. 'KernelFunction' calls the three different
    %types of kernels used in the function as defined by the for loop
    %('linear', 'rbf' for Gaussian, and 'polynomial' of order 3);
    %'Standardize' normalizes all tof the parameters so that specifically
    %larger parameters are not overweighted. 
    Class_Data=fitcsvm(Flattened_Cones_Params,Flattened_Cones_Labels,'KernelFunction',Kernel_Func,'Standardize',true, 'BoxConstraint', 1);
    %Define minfn as the function that minimizes the kfoldLoss by modfiying
    minfn=@(z)kfoldLoss(fitcsvm(Flattened_Cones_Params,Flattened_Cones_Labels,'CrossVal', 'on', 'KernelFunction',Kernel_Func,'Standardize',true, 'BoxConstraint', exp(z(1)), 'KernelScale', exp(z(2))));
    %the box constraint and Kernal Scale parameters
    %Loosen the tolerance for use in the function
    opts=optimset('TolX',5,'TolFun',5);
    %By changing the 
    m=2;
    fval=zeros(m,1);
    z=ones(1,2);
    z_mat=ones(m,2);
    for j=1:m
        [searchmin, fval(j)]=fminsearch(minfn,randn(2,1),opts);
        z=exp(searchmin);
        z_mat(j,:)=z;
    end
    
    z=z_mat(fval == min(fval),:);
    Class_Data=fitcsvm(Flattened_Cones_Params,Flattened_Cones_Labels,'KernelFunction',Kernel_Func,'Standardize',true, 'BoxConstraint', z(1),'KernelScale',z(2));
    [~, Y_predict(:,:,i)]=predict(Class_Data,Flattened_Cones_Params);
    Class_Data_Posterior=fitSVMPosterior(Class_Data);
    [~, Post_predict(:,:,i)]=predict(Class_Data_Posterior,Flattened_Cones_Params);
    %Initialize elements of confusion matrix and ROC curve
    Num_TP=0; Num_TN=0; Num_FP=0; Num_FN=0;

    for j=1:length(Flattened_Cones_Labels)
        %Define a True Positive (TP) as a setos that is classified as a
        %setosa
        if Y_predict(j,i)==1 && Y_log(j)==1
            Num_TP=Num_TP+1;
        %Define a True Negative (TN) as other irises that are classified
        %as other irises
        elseif Y_predict(j,i)==0 && Y_log(j)==0
            Num_TN=Num_TN+1;
        %Define a False Positive (FP) as another iris that is classified as a
        %setosa
        elseif Y_predict(j,i)==1 && Y_log(j)==0
            Num_FP=Num_FP+1;
        %Define a False Negative (FN) as a setosa that is classified as
        %another iris
        elseif Y_predict(j,i)==0 && Y_log(j)==1
            Num_FN=Num_FN+1;
        end

    end
    %Define sensitivity
    Se(i)=Num_TP/(Num_TP+Num_FN);
    %Define specificity
    Sp(i)=Num_TN/(Num_TN+Num_FP);
    %Define the false positive rate as 1.-Sp
    FP_rate(i)=1.-Sp(i);

    %Plot an ROC curve and label it in the legend 
    %Transpose for ease of use
    Y_working=Y_predict(:,i)';
    Y_working2=Post_predict(:,i)';
    %targets is a 2x150 matrix that represents a 1 in the row of whichever
    %category the number represents and a 0 in the other row
    targets(1,Y_working==1)=1;
    targets(2,Y_working==1)=0;
    targets(1,Y_working==-1)=0;
    targets(2,Y_working==-1)=1;
    %outputs is a 2x150 matrix that represents the posterior probability
    %for each category
    outputs(1,Y_working2==1)=1;
    outputs(2,Y_working2==1)=0;
    outputs(1,Y_working2==-1)=0;
    outputs(2,Y_working2==-1)=1;
    [tpr(i), fpr(i), thresholds(i)]=roc(targets, outputs);
    plotroc
    %Define a 2x2 confusion matrix that compares the number of TP, TN, FP, and
    %FN
    Conf_matrix(:,:,i)=[Num_TP, Num_FN; Num_FP, Num_TN];

end
