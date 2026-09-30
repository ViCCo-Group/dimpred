% dimpred: predict the SPoSE dimensions of human object representations for any image
%
% DimPred predicts, for any image, its values on the SPoSE dimensions of
% human mental object representations (49 dimensions in Hebart et al.,
% 2020, or 66 in Hebart et al., 2023). A network (the image encoder of
% CLIP) turns each image into a feature vector, and one linear regression
% per dimension, trained on the 1854 THINGS reference images, maps these
% features to dimension values. From the predicted dimensions, we can then
% compute the predicted perceived similarity between images.
%
% The MATLAB functions have the names of the Python functions with the
% prefix dimpred_ and give the same numbers. They use the model files in
% ../dimpred/models, so this folder has to stay in the dimpred repository.
%
% Functions:
%   dimpred_list_models       names of the models that come with dimpred
%   dimpred_load_model        load a model (weights, feature mean and scale, dimension means, labels, info)
%   dimpred_find_images       image files in a folder, sorted by name
%   dimpred_extract_features  network features of images (runs Python with torch and open_clip)
%   dimpred_predict           predicted dimension values from network features
%   dimpred_similarity        predicted similarity between images from their predicted dimensions
%
% Example:
%   setenv('DIMPRED_PYTHON', '/path/to/python')  % Python for dimpred_extract_features
%   files = dimpred_find_images('my_images');
%   features = dimpred_extract_features(files);  % default model: vitb32_66d_elastic
%   embedding = dimpred_predict(features);       % n_images x 66
%   S = dimpred_similarity(embedding);           % n_images x n_images
%
% Reference: Kaniuth, P., Mahner, F. P., Perkuhn, J., & Hebart, M. N.
% (2025). A high-throughput approach for the efficient prediction of
% perceived similarity of natural objects. eLife 14:RP105394.
%
% Martin Hebart, 2026/09/30
