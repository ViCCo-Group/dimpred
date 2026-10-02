% function tests = test_dimpred_extract_features
%
% Tests for dimpred_extract_features that really run the networks. The
% function runs the Python command line tool of dimpred
% (python -m dimpred ... --features-only) to get the network features of
% images.
%
% We extract the features of the 3 CC0 images in tests/fixtures/images and
% compare them with reference features that were extracted separately with
% open_clip and, for AligNet, with the TensorFlow model of its authors
% (tests/fixtures/make_fixtures.py). Different devices (cpu, gpu) give tiny
% differences, so we require a correlation > 0.99999 and a maximum absolute
% difference < 2e-3 for each image (1e-4 for AligNet). These tests catch two
% bugs of earlier versions:
%   - using open_clip's plain ViT-B-32, which uses GELU instead of the
%     QuickGELU of OpenAI's CLIP. The weights are the same, but the
%     features correlate only ~0.98 with the correct ones.
%   - sorting the file names instead of keeping the given order, which
%     pairs images with the wrong rows
%
% All tests here need Python with torch and open_clip. Set the environment
% variable DIMPRED_PYTHON to that Python, e.g. in MATLAB
%   setenv('DIMPRED_PYTHON', '/path/to/python')
% Without it, these tests are skipped (with the reason). The test of the
% default model (AligNet SigLIP2-B) also needs its weights, which the tests
% never download: set DIMPRED_ALIGNET_WEIGHTS to alignet_siglip2_b.safetensors
% (or have them in ~/.cache/dimpred), otherwise it is skipped. Most tests
% use vitb32_66d_elastic, which is small. They take a few minutes, mostly
% for loading the networks (10 Python runs, 3 of them with the large
% RN50x64). run_dimpred_tests('fast') leaves this file out. The
% tests that need no network (wrong input, failing Python, the command
% that is sent to Python) are in test_dimpred_extract_features_errors.m
% and always run.
%
% Run with
%   runtests('test_dimpred_extract_features')
% or all MATLAB tests with run_dimpred_tests.
%
% Hebartlab, 2026/09/30
%
% See also DIMPRED_EXTRACT_FEATURES, TEST_DIMPRED_EXTRACT_FEATURES_ERRORS,
%   RUN_DIMPRED_TESTS

% History:
% 2026/10/02: the default model is alignet_siglip2b_66d_ridge (AligNet
%   SigLIP2-B); the tests that used the default model use vitb32_66d_elastic
% 2026/09/30: after review: order test with a repeated image, file names
%   with spaces and quotes, cfg.batch_size and cfg.device, model as struct
%   and as file, unreadable image. Folders are no longer accepted (as in
%   Python), the error tests moved to test_dimpred_extract_features_errors.
% 2026/09/30: written before the code (test-driven development)

function tests = test_dimpred_extract_features
tests = functiontests(localfunctions);
end


%% Setup

function setupOnce(testCase)

% Add the dimpred MATLAB functions to the path (restored in teardownOnce)
here = fileparts(mfilename('fullpath')); % .../dimpred/tests/matlab
repo = fileparts(fileparts(here));
testCase.TestData.old_path = addpath(fullfile(repo, 'matlab'));

% Load the reference numbers (made by tests/fixtures/make_fixtures.py)
fixture_folder = fullfile(repo, 'tests', 'fixtures');
fixture_file = fullfile(fixture_folder, 'reference_data.mat');
testCase.assertTrue(exist(fixture_file, 'file') == 2, ...
    sprintf('Fixture file %s not found. It is made by tests/fixtures/make_fixtures.py.', fixture_file));
ref = load(fixture_file, 'cc0_files', 'cc0_features_rn50x64', 'cc0_features_vitb32', 'cc0_features_alignet', ...
    'cc0_published_rn50x64_49d_ridge');
testCase.TestData.ref = ref;
testCase.TestData.image_folder = fullfile(fixture_folder, 'images');
testCase.TestData.cc0_images = fullfile(testCase.TestData.image_folder, ref.cc0_files); % in the order of the fixture

% Check whether we have a Python that can run the networks. We only check
% torch and open_clip here, finding the dimpred package is the job of
% dimpred_extract_features.
python = getenv('DIMPRED_PYTHON');
python_ok = false;
python_problem = 'the environment variable DIMPRED_PYTHON is not set';
if ~isempty(python)
    [status, output] = system(sprintf('"%s" -c "import torch, open_clip"', python));
    python_ok = status == 0;
    if ~python_ok
        python_problem = sprintf('%s cannot import torch and open_clip (%s)', python, strtrim(output));
    end
end
testCase.TestData.python = python;
testCase.TestData.python_ok = python_ok;
testCase.TestData.python_problem = python_problem;

% The AligNet weights: the file in DIMPRED_ALIGNET_WEIGHTS, or the one that
% dimpred downloaded before (as in Python, see dimpred/alignet.py)
weights = getenv('DIMPRED_ALIGNET_WEIGHTS');
if isempty(weights)
    home = getenv('HOME');
    if isempty(home), home = getenv('USERPROFILE'); end % Windows
    weights = fullfile(home, '.cache', 'dimpred', 'alignet_siglip2_b.safetensors');
end
testCase.TestData.alignet_ok = exist(weights, 'file') == 2;

% Extracted features, shared by the tests of this file (see extract_once).
% containers.Map is a handle object, so what one test stores, the others see.
testCase.TestData.extracted = containers.Map;

end

function teardownOnce(testCase)
path(testCase.TestData.old_path);
end


%% Features

function test_default_model_features_match_reference(testCase)
% Only the images are given: the default model (alignet_siglip2b_66d_ridge,
% network AligNet SigLIP2-B) and the Python in DIMPRED_PYTHON are used. The
% reference features come from the TensorFlow model of the AligNet authors.
testCase.assumeTrue(testCase.TestData.alignet_ok, ['Skipped because the AligNet weights were not found. ' ...
    'Set DIMPRED_ALIGNET_WEIGHTS to alignet_siglip2_b.safetensors to run it.']);
ref = testCase.TestData.ref;
features = extract_once(testCase, testCase.TestData.cc0_images);
verify_same_features(testCase, features, ref.cc0_features_alignet, ref.cc0_files, 'AligNet SigLIP2-B', 1e-4);
end

function test_vit_features_match_reference(testCase)
% With open_clip's plain ViT-B-32 the correlation with the reference is
% ~0.98, which fails here
ref = testCase.TestData.ref;
features = extract_once(testCase, testCase.TestData.cc0_images, 'vitb32_66d_elastic');
verify_same_features(testCase, features, ref.cc0_features_vitb32, ref.cc0_files, 'ViT-B-32-quickgelu');
end

function test_rn50x64_features_match_reference(testCase)
ref = testCase.TestData.ref;
features = extract_once(testCase, testCase.TestData.cc0_images, 'rn50x64_49d_ridge', python_cfg(testCase));
verify_same_features(testCase, features, ref.cc0_features_rn50x64, ref.cc0_files, 'RN50x64');
end

function test_features_are_single_with_one_row_per_image(testCase)
% As in Python (float32), the features are single, images x features
features = extract_once(testCase, testCase.TestData.cc0_images, 'vitb32_66d_elastic');
testCase.verifyClass(features, 'single', 'The features should be single (float32), as in Python');
testCase.verifySize(features, [3 512], '3 images with ViT-B-32-quickgelu should give 3 x 512 features');
end

function test_predictions_from_extracted_features_match_published(testCase)
% From the images to Philipp's published predictions (his own extraction),
% the whole way a user goes
ref = testCase.TestData.ref;
features = extract_once(testCase, testCase.TestData.cc0_images, 'rn50x64_49d_ridge', python_cfg(testCase));
embedding = dimpred_predict(features, 'rn50x64_49d_ridge');
testCase.verifyEqual(embedding, ref.cc0_published_rn50x64_49d_ridge, 'AbsTol', 2e-3, ...
    'Predictions from extracted features differ from Philipp''s published predictions of the CC0 images');
end


%% Order of the images

function test_rows_follow_given_order(testCase)
% The rows have to be in the order of the given files, never sorted
ref = testCase.TestData.ref;
[images, rows] = unsorted_images_with_repeat(testCase);
features = extract_once(testCase, images, 'vitb32_66d_elastic', python_cfg(testCase));
verify_same_features(testCase, features, ref.cc0_features_vitb32(rows, :), file_names(images), 'ViT-B-32-quickgelu');
end

function test_repeated_image_gives_repeated_row(testCase)
% An image that is given twice gets two rows, so that row i always belongs
% to file i. This fails if the files are made unique on the way.
images = unsorted_images_with_repeat(testCase);
features = extract_once(testCase, images, 'vitb32_66d_elastic', python_cfg(testCase));
testCase.verifyEqual(size(features, 1), 4, ...
    sprintf('4 files (one of them twice) should give 4 rows, got %i', size(features, 1)));
end

function test_second_output_lists_given_files(testCase)
% The second output says which row is which image: the given files, in the
% given order, with repeats
images = unsorted_images_with_repeat(testCase);
[~, files] = extract_once(testCase, images, 'vitb32_66d_elastic', python_cfg(testCase));
testCase.assertSize(files, [4 1], 'The second output should be a cell column with one file per row');
testCase.verifyEqual(file_names(files), file_names(images), 'The second output should list the files in the given order');
end

function test_single_file_as_text(testCase)
% One image can be given as text instead of a cell
ref = testCase.TestData.ref;
[features, files] = extract_once(testCase, testCase.TestData.cc0_images{2}, 'vitb32_66d_elastic', python_cfg(testCase));
verify_same_features(testCase, features, ref.cc0_features_vitb32(2, :), ref.cc0_files(2), 'ViT-B-32-quickgelu');
testCase.verifySize(files, [1 1], 'For one file given as text, the second output should be a 1 x 1 cell');
end

function test_paths_with_spaces_and_quotes(testCase)
% dimpred_extract_features builds a command line for Python, so folders
% and file names with spaces, quotes or brackets have to arrive as they
% are. We copy the CC0 images to such names.
cfg = python_cfg(testCase); % first, so that the test is skipped without Python
ref = testCase.TestData.ref;
new_names = {'first image.jpg'; 'it''s the second.jpg'; 'third (copy) & more.jpg'};
folder = fullfile(make_temp_folder(testCase), 'my images');
mkdir(folder);
images = fullfile(folder, new_names);
for i_image = 1:3
    copyfile(testCase.TestData.cc0_images{i_image}, images{i_image});
end
features = extract_once(testCase, images, 'vitb32_66d_elastic', cfg);
verify_same_features(testCase, features, ref.cc0_features_vitb32, new_names, 'ViT-B-32-quickgelu');
end


%% Options and ways to give the model

function test_batch_size_and_device_options(testCase)
% cfg.batch_size and cfg.device are passed on to Python. With 3 images and
% a batch size of 2, the last batch has only one image. The features must
% not change.
ref = testCase.TestData.ref;
cfg = python_cfg(testCase);
cfg.batch_size = 2;
cfg.device = 'cpu';
features = extract_once(testCase, testCase.TestData.cc0_images, 'vitb32_66d_elastic', cfg);
verify_same_features(testCase, features, ref.cc0_features_vitb32, ref.cc0_files, 'ViT-B-32-quickgelu (batch size 2, cpu)');
end

function test_model_given_as_struct(testCase)
% A loaded model defines the network as well. We use an RN50x64 model, so
% that ignoring the struct (and using the network of the default model)
% would give features of the wrong size.
cfg = python_cfg(testCase); % first, so that the test is skipped without Python
ref = testCase.TestData.ref;
model = dimpred_load_model('rn50x64_49d_ridge');
features = extract_once(testCase, testCase.TestData.cc0_images(2), model, cfg);
verify_same_features(testCase, features, ref.cc0_features_rn50x64(2, :), ref.cc0_files(2), 'RN50x64 (model given as struct)');
end

function test_model_given_as_file(testCase)
% A model file somewhere else, here in a folder with a space in its name
cfg = python_cfg(testCase); % first, so that the test is skipped without Python
ref = testCase.TestData.ref;
original = dimpred_load_model('rn50x64_49d_ridge');
folder = fullfile(make_temp_folder(testCase), 'my models');
mkdir(folder);
model_file = fullfile(folder, 'my model.mat');
copyfile(original.file, model_file);
features = extract_once(testCase, testCase.TestData.cc0_images(3), model_file, cfg);
verify_same_features(testCase, features, ref.cc0_features_rn50x64(3, :), ref.cc0_files(3), 'RN50x64 (model given as file)');
end


%% Errors from Python

function test_unreadable_image_gives_python_error(testCase)
% An image file that exists but cannot be read passes the check in MATLAB
% and makes Python fail. The user should get an error that shows Python's
% message (which names the image), not empty or old features.
cfg = python_cfg(testCase);
broken = fullfile(make_temp_folder(testCase), 'broken.jpg');
fid = fopen(broken, 'w'); % empty file
fclose(fid);
images = [testCase.TestData.cc0_images(1); {broken}];
err = error_of(@() dimpred_extract_features(images, 'vitb32_66d_elastic', cfg));
testCase.assertNotEmpty(err, 'An image that cannot be read should give an error');
testCase.verifyEqual(err.identifier, 'dimpred:pythonFailed', ['Wrong error: ' err.message]);
testCase.verifySubstring(err.message, 'broken.jpg', ...
    'The error message should include Python''s message, which names the image that could not be read');
end


%% Helpers

function [images, rows] = unsorted_images_with_repeat(testCase)
% The CC0 images in the order 3 1 3 2 of the sorted names, which is not
% sorted and has one image twice. We start from the sorted names, so that
% this holds whatever the order of cc0_files is. rows are the rows of these
% images in the reference features.
ref = testCase.TestData.ref;
names = sort(ref.cc0_files);
names = names([3 1 3 2]);
[~, rows] = ismember(names, ref.cc0_files);
images = fullfile(testCase.TestData.image_folder, names);
end

function cfg = python_cfg(testCase)
% cfg with the Python of DIMPRED_PYTHON. Skips the test if there is no
% usable Python.
skip_without_python(testCase);
cfg.python = testCase.TestData.python;
end

function skip_without_python(testCase)
testCase.assumeTrue(testCase.TestData.python_ok, ...
    sprintf('Skipped because %s. Set DIMPRED_PYTHON to a Python with torch and open_clip to run it.', ...
    testCase.TestData.python_problem));
end

function [features, files] = extract_once(testCase, varargin)
% Calls dimpred_extract_features(varargin{:}) and keeps the result for the
% other tests. Each call starts Python and loads the network, which takes
% a while (RN50x64 is large), so tests with the same input share one run.
% Skips the test if there is no usable Python.
skip_without_python(testCase);
key = strjoin(cellfun(@describe, varargin, 'UniformOutput', false), ' ; ');
if ~isKey(testCase.TestData.extracted, key)
    [features, files] = dimpred_extract_features(varargin{:});
    testCase.TestData.extracted(key) = {features, files};
end
result = testCase.TestData.extracted(key);
[features, files] = result{:};
end

function key = describe(value)
% Text that identifies one input of dimpred_extract_features (for extract_once)
if ischar(value)
    key = value;
elseif iscell(value)
    % in braces, because one file in a cell is not the same input as text
    key = ['{' strjoin(cellfun(@describe, value(:)', 'UniformOutput', false), ', ') '}'];
elseif isempty(value)
    key = '[]';
elseif isnumeric(value)
    key = num2str(value);
elseif isstruct(value) && isfield(value, 'weights') % a loaded model
    key = ['model struct ' value.info.name];
elseif isstruct(value) % cfg
    fields = fieldnames(value);
    parts = cellfun(@(f) [f '=' describe(value.(f))], fields(:)', 'UniformOutput', false);
    key = ['cfg(' strjoin(parts, ', ') ')'];
else
    error('describe cannot handle input of class %s', class(value))
end
end

function verify_same_features(testCase, features, reference, names, network, max_difference)
% Per image: correlation > 0.99999 and maximum absolute difference <
% max_difference (default: 2e-3)
if nargin < 6, max_difference = 2e-3; end
testCase.assertSize(features, size(reference), sprintf('%s features should be images x features', network));
for i_image = 1:numel(names)
    extracted = double(features(i_image, :));
    r = corrcoef(extracted, reference(i_image, :));
    testCase.verifyGreaterThan(r(1, 2), 0.99999, ...
        sprintf('%s features of %s (row %i) correlate only r = %.6f with the reference', network, names{i_image}, i_image, r(1, 2)));
    testCase.verifyLessThan(max(abs(extracted - reference(i_image, :))), max_difference, ...
        sprintf('%s features of %s (row %i) differ from the reference by more than %g', network, names{i_image}, ...
        i_image, max_difference));
end
end

function names = file_names(files)
% File names without folder, as a cell column
[~, base, ext] = cellfun(@fileparts, files(:), 'UniformOutput', false);
names = strcat(base, ext);
end

function folder = make_temp_folder(testCase)
% Temporary folder that is deleted after the test
folder = tempname;
mkdir(folder);
testCase.addTeardown(@rmdir, folder, 's');
end

function err = error_of(f)
% The error that f() gives, or [] if it gives none
err = [];
try
    f();
catch err
end
end
