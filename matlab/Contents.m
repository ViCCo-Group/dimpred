% dimpred: predict the SPoSE dimensions of human object representations for any image
%
% DimPred predicts, for any image, its values on the SPoSE dimensions of
% human mental object representations (49 dimensions in Hebart et al.,
% 2020, or 66 in Hebart et al., 2023). A network (by default the image
% encoder of AligNet SigLIP2-B, Muttenthaler et al., 2025) turns each image
% into a feature vector, and one linear regression per dimension, trained
% on the 1854 THINGS reference images, maps these features to dimension
% values. From the predicted dimensions, we can then compute the predicted
% perceived similarity between images. Since 1.3.0, the models
% ..._36d_... predict the 36 dimensions of human material perception of
% the STUFF dataset (Schmidt et al., 2025); their similarity is
% dimpred_similarity(embedding, '2afc').
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
%                             (and, for the close pairs, their features)
%   dimpred_rise              heatmaps: which parts of an image drive its predicted dimensions (runs Python)
%
% Example:
%   setenv('DIMPRED_PYTHON', '/path/to/python')  % Python for dimpred_extract_features and dimpred_rise
%   files = dimpred_find_images('my_images');
%   features = dimpred_extract_features(files);  % default model: alignet_siglip2b_66d_kernel
%   embedding = dimpred_predict(features);       % n_images x 66
%   S = dimpred_similarity(embedding, [], features);  % n_images x n_images, with the close pairs
%
% Reference: Kaniuth, P., Mahner, F. P., Perkuhn, J., & Hebart, M. N.
% (2025). A high-throughput approach for the efficient prediction of
% perceived similarity of natural objects. eLife 14:RP105394.
% For the AligNet models (alignet_*, e.g. the default model), please also
% cite AligNet: Muttenthaler, L., et al. (2025). Aligning machine and human
% visual representations across abstraction levels. Nature 647, 349-355.
% For the material models (..._36d_...), please also cite the paper of
% their dimensions: Schmidt, F., Hebart, M. N., Schmid, A. C., & Fleming,
% R. W. (2025). Core dimensions of human material perception. PNAS 122(10),
% e2417202122.
%
% Hebartlab, 2026/09/30

% History:
% 2026/10/26: the material models (dimpred 1.3.0) and their citation; the
%   AligNet citation for all AligNet models
% 2026/10/04: new default model alignet_siglip2b_66d_kernel; dimpred_similarity
%   with the close pairs (features)
% 2026/10/02: dimpred_rise, new default model alignet_siglip2b_66d_ridge
% 2026/09/30: written for the first release of the package
