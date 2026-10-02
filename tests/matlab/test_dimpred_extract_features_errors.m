% function tests = test_dimpred_extract_features_errors
%
% Tests for dimpred_extract_features that need no network and run in a
% few seconds: wrong input, a Python that fails, and the command line that
% is sent to Python. The tests that really extract features are in
% test_dimpred_extract_features.m.
%
% dimpred_extract_features checks the image files in MATLAB before it
% starts Python. It accepts files only, no folders (use dimpred_find_images
% for a folder), so that row i of the features always belongs to file i.
% If Python fails, it gives an error that includes what Python printed.
%
% Building the command line is the only thing the MATLAB function does
% that the Python code does not, so we test it without Python. We use a
% fake "python": a small shell script that writes the arguments it gets to
% a text file, prints an error text and exits with status 1.
% dimpred_extract_features then has to give the error dimpred:pythonFailed,
% and we can read which arguments arrived after "-m dimpred". The file
% names have to arrive in the given order (with repeats), each as one
% argument, also with spaces and quotes, and the options have to be the
% ones of the command line tool (--model, --batch-size, --device,
% --features-only). Python is cfg.python, or else the environment
% variable DIMPRED_PYTHON. A shell script does not run on Windows, so the
% tests with the fake Python are skipped there. The image files here are
% empty, because Python never reads them.
%
% For many images, the command line would get too long, so Python is
% started several times, and the parts have to be put together in the
% right order. For this test, a second fake Python saves one number per
% image as its feature, so that we can see which image each row belongs
% to. It needs a real Python with scipy to write the .mat file, and is
% skipped if the environment variable DIMPRED_PYTHON is not set.
%
% Run with
%   runtests('test_dimpred_extract_features_errors')
% or all MATLAB tests with run_dimpred_tests (also in 'fast' mode).
%
% Hebartlab, 2026/09/30
%
% See also DIMPRED_EXTRACT_FEATURES, TEST_DIMPRED_EXTRACT_FEATURES,
%   RUN_DIMPRED_TESTS

% History:
% 2026/09/30: after the second review: many images with several Python runs
% 2026/09/30: written after review, the error tests came from
%   test_dimpred_extract_features.m (so that they also run in 'fast' mode),
%   the tests with the fake Python are new

function tests = test_dimpred_extract_features_errors
tests = functiontests(localfunctions);
end


%% Setup

function setupOnce(testCase)

% Add the dimpred MATLAB functions to the path (restored in teardownOnce)
here = fileparts(mfilename('fullpath')); % .../dimpred/tests/matlab
repo = fileparts(fileparts(here));
testCase.TestData.old_path = addpath(fullfile(repo, 'matlab'));

% A Python that does not exist
testCase.TestData.no_python = fullfile(tempname, 'python');

end

function teardownOnce(testCase)
path(testCase.TestData.old_path);
end


%% Wrong input (checked before Python starts)

function test_missing_image_gives_error(testCase)
% The files are checked in MATLAB before Python is started. The Python
% given here does not exist, so dimpred:pythonFailed would show that the
% files were not checked first.
images = make_image_files(testCase, {'a.jpg'});
images{2, 1} = fullfile(tempname, 'missing.jpg');
cfg.python = testCase.TestData.no_python;
err = error_of(@() dimpred_extract_features(images, [], cfg));
testCase.assertNotEmpty(err, 'A missing image file should give an error');
testCase.verifyEqual(err.identifier, 'dimpred:fileNotFound', ...
    ['A missing image file should give dimpred:fileNotFound before Python is started, got: ' err.message]);
testCase.verifySubstring(err.message, 'missing.jpg', 'The error message should name the missing file');
end

function test_folder_gives_error(testCase)
% Folders are not accepted, as in Python: the rows of the features must
% always be in the order of the given files. The message should tell the
% user to list the folder with dimpred_find_images.
images = make_image_files(testCase, {'a.jpg'});
folder = fileparts(images{1});
cfg.python = testCase.TestData.no_python;
err = error_of(@() dimpred_extract_features(folder, [], cfg));
testCase.assertNotEmpty(err, 'A folder instead of image files should give an error');
testCase.verifyEqual(err.identifier, 'dimpred:fileNotFound', ...
    ['A folder instead of image files should give dimpred:fileNotFound, got: ' err.message]);
testCase.verifySubstring(err.message, 'dimpred_find_images', ...
    'For a folder, the error message should point to dimpred_find_images');
end


%% Python fails

function test_python_that_does_not_exist_gives_error(testCase)
% The user should get an error, not empty or old features
images = make_image_files(testCase, {'a.jpg'});
cfg.python = testCase.TestData.no_python;
verify_python_fails(testCase, @() dimpred_extract_features(images, [], cfg));
end

function test_python_error_text_is_shown(testCase)
% When Python starts and then fails (exit status 1), the user has to see
% what Python printed, e.g. which image could not be read
fake = make_fake_python(testCase, 'fake');
images = make_image_files(testCase, {'a.jpg'});
cfg.python = fake.python;
err = verify_python_fails(testCase, @() dimpred_extract_features(images, [], cfg));
testCase.verifySubstring(err.message, fake.error_text, 'The error message should include the text that Python printed');
end


%% Command line sent to Python (with the fake Python)

function test_images_are_passed_in_given_order(testCase)
% The order is not sorted and one image comes twice. This fails if the
% files are sorted or made unique before they are passed to Python.
fake = make_fake_python(testCase, 'fake');
images = make_image_files(testCase, {'c.jpg'; 'a.jpg'; 'c.jpg'; 'b.jpg'});
cfg.python = fake.python;
verify_python_fails(testCase, @() dimpred_extract_features(images, [], cfg));
args = arguments_after_dimpred(testCase, fake);
testCase.verifyEqual(args(ismember(args, images)), images, ...
    'The image files should be passed to Python in the given order, with repeats');
end

function test_names_with_spaces_and_quotes_are_passed_unchanged(testCase)
% Each file has to arrive as one argument, exactly as given. This fails if
% the paths are not quoted in the command line.
fake = make_fake_python(testCase, 'fake');
images = make_image_files(testCase, {'first image.jpg'; 'it''s the second.jpg'; 'third (copy) & more.jpg'}, 'my images');
cfg.python = fake.python;
verify_python_fails(testCase, @() dimpred_extract_features(images, [], cfg));
args = arguments_after_dimpred(testCase, fake);
for i_image = 1:numel(images)
    testCase.verifyTrue(any(strcmp(args, images{i_image})), ...
        sprintf('The file %s should arrive in Python as one argument. Arguments were:\n%s', images{i_image}, strjoin(args', '\n')));
end
end

function test_options_are_passed(testCase)
% The model, cfg.batch_size and cfg.device have to reach the command line
% tool with its option names, and the tool has to be asked for features
% only. A misspelled option (e.g. --batch_size) would break every call.
fake = make_fake_python(testCase, 'fake');
images = make_image_files(testCase, {'a.jpg'});
cfg.python = fake.python;
cfg.batch_size = 2;
cfg.device = 'cpu';
verify_python_fails(testCase, @() dimpred_extract_features(images, 'rn50x64_49d_ridge', cfg));
args = arguments_after_dimpred(testCase, fake);
all_args = strjoin(args', ' ');

testCase.verifyTrue(any(strcmp(args, '--features-only')), ['--features-only is missing: ' all_args]);
testCase.verifyEqual(option_value(args, '--batch-size'), '2', ['cfg.batch_size = 2 should give --batch-size 2: ' all_args]);
testCase.verifyEqual(option_value(args, '--device'), 'cpu', ['cfg.device = ''cpu'' should give --device cpu: ' all_args]);
% The model can be passed by name or as the path of its file
[~, model_name] = fileparts(option_value(args, '--model'));
testCase.verifyEqual(model_name, 'rn50x64_49d_ridge', ['The model should be passed with --model: ' all_args]);
end


%% Which Python is used

function test_python_path_with_space(testCase)
fake = make_fake_python(testCase, 'my python');
images = make_image_files(testCase, {'a.jpg'});
cfg.python = fake.python;
verify_python_fails(testCase, @() dimpred_extract_features(images, [], cfg));
testCase.verifyNotEmpty(recorded_calls(fake), ...
    sprintf('The Python %s was not started. A path with a space has to be quoted in the command.', fake.python));
end

function test_dimpred_python_is_used_without_cfg(testCase)
fake = make_fake_python(testCase, 'fake');
set_dimpred_python(testCase, fake.python);
images = make_image_files(testCase, {'a.jpg'});
verify_python_fails(testCase, @() dimpred_extract_features(images));
testCase.verifyNotEmpty(recorded_calls(fake), ...
    'Without cfg, dimpred_extract_features should run the Python given in DIMPRED_PYTHON');
end

function test_dimpred_python_is_used_without_cfg_python(testCase)
% cfg with other fields, but without python
fake = make_fake_python(testCase, 'fake');
set_dimpred_python(testCase, fake.python);
images = make_image_files(testCase, {'a.jpg'});
cfg.device = 'cpu';
verify_python_fails(testCase, @() dimpred_extract_features(images, [], cfg));
testCase.verifyNotEmpty(recorded_calls(fake), ...
    'Without cfg.python, dimpred_extract_features should run the Python given in DIMPRED_PYTHON');
end

function test_cfg_python_comes_before_dimpred_python(testCase)
fake = make_fake_python(testCase, 'fake');
set_dimpred_python(testCase, testCase.TestData.no_python);
images = make_image_files(testCase, {'a.jpg'});
cfg.python = fake.python;
verify_python_fails(testCase, @() dimpred_extract_features(images, [], cfg));
testCase.verifyNotEmpty(recorded_calls(fake), 'cfg.python should be used, not DIMPRED_PYTHON');
end


%% Many images (several Python runs)

function test_many_images_keep_their_order_across_python_runs(testCase)
% 1500 images with names of 129 characters do not fit into one command
% line, so Python is started several times (about 4 times on macOS and
% Linux). The feature of each image is the number in its name, so the
% features have to be 1, 2, ..., 1500. This fails if the parts are put
% together in the wrong order.
python = getenv('DIMPRED_PYTHON');
testCase.assumeNotEmpty(python, 'The fake dimpred needs a real Python with scipy, set DIMPRED_PYTHON to run this test');
fake = make_numbering_python(testCase, python);
n_images = 1500;
names = arrayfun(@(i) sprintf('%s_%04i.jpg', repmat('x', 1, 120), i), (1:n_images)', 'UniformOutput', false);
images = make_image_files(testCase, names);
cfg.python = fake.python;
features = dimpred_extract_features(images, [], cfg);
n_calls = numel(splitlines(strtrim(fileread(fake.log_file))));
testCase.assertGreaterThan(n_calls, 1, 'For 1500 long file names, Python should be started more than once');
testCase.verifyEqual(double(features), (1:n_images)', ...
    sprintf('The rows of the features should be in the order of the images, also across %i Python runs', n_calls));
end


%% Helpers

function fake = make_fake_python(testCase, folder_name)
% Shell script in a new folder of the given name that stands in for
% Python. It adds its arguments to fake.log_file (one per line, each call
% ends with fake.end_marker), prints fake.error_text and exits with status 1.
testCase.assumeFalse(ispc, 'The fake Python is a shell script, which does not run on Windows');
folder = fullfile(make_temp_folder(testCase), folder_name);
mkdir(folder);
fake.python = fullfile(folder, 'python');
fake.log_file = fullfile(folder, 'calls.txt');
fake.end_marker = '=== end of call ===';
fake.error_text = 'fake python stopped here on purpose';
fid = fopen(fake.python, 'w');
testCase.assertGreaterThan(fid, 0, sprintf('Could not create %s', fake.python));
fprintf(fid, '#!/bin/sh\n');
fprintf(fid, 'for arg in "$@"; do printf ''%%s\\n'' "$arg"; done >> ''%s''\n', fake.log_file);
fprintf(fid, 'echo ''%s'' >> ''%s''\n', fake.end_marker, fake.log_file);
fprintf(fid, 'echo ''%s'' >&2\n', fake.error_text);
fprintf(fid, 'exit 1\n');
fclose(fid);
fileattrib(fake.python, '+x');
end

function fake = make_numbering_python(testCase, python)
% Shell script in a new folder that stands in for Python. It runs
% fake_dimpred.py with the given real Python instead of "-m dimpred" and
% adds a line to fake.log_file for each call. fake_dimpred.py saves, as the
% only feature of each image, the 4 digits before the extension of its
% name (img_0012.jpg gives 12), in the file given with --out.
testCase.assumeFalse(ispc, 'The fake Python is a shell script, which does not run on Windows');
folder = make_temp_folder(testCase);
script = fullfile(folder, 'fake_dimpred.py');
fake.python = fullfile(folder, 'python');
fake.log_file = fullfile(folder, 'calls.txt');
fid = fopen(script, 'w');
testCase.assertGreaterThan(fid, 0, sprintf('Could not create %s', script));
fprintf(fid, 'import sys\n');
fprintf(fid, 'import scipy.io\n');
fprintf(fid, 'args = sys.argv[3:]  # the arguments after -m dimpred\n');
fprintf(fid, 'images = []\n');
fprintf(fid, 'for arg in args:\n');
fprintf(fid, '    if arg.startswith("--"):\n');
fprintf(fid, '        break\n');
fprintf(fid, '    images.append(arg)\n');
fprintf(fid, 'out = args[args.index("--out") + 1]\n');
fprintf(fid, 'scipy.io.savemat(out, {"features": [[float(fname[-8:-4])] for fname in images]})\n');
fclose(fid);
fid = fopen(fake.python, 'w');
testCase.assertGreaterThan(fid, 0, sprintf('Could not create %s', fake.python));
fprintf(fid, '#!/bin/sh\n');
fprintf(fid, 'echo call >> ''%s''\n', fake.log_file);
fprintf(fid, 'exec ''%s'' ''%s'' "$@"\n', python, script);
fclose(fid);
fileattrib(fake.python, '+x');
end

function calls = recorded_calls(fake)
% Arguments of each call of the fake Python, one cell column per call
calls = {};
if exist(fake.log_file, 'file') ~= 2
    return % not called
end
lines = splitlines(fileread(fake.log_file));
current = {};
for i_line = 1:numel(lines)
    if strcmp(lines{i_line}, fake.end_marker)
        calls{end+1, 1} = current; %#ok<AGROW>
        current = {};
    else
        current{end+1, 1} = lines{i_line}; %#ok<AGROW>
    end
end
end

function args = arguments_after_dimpred(testCase, fake)
% Arguments that the fake Python got after "-m dimpred", as a cell column.
% Stops the test if Python was never called with -m dimpred.
calls = recorded_calls(fake);
for i_call = 1:numel(calls)
    call = calls{i_call};
    i_m = find(strcmp(call(1:end-1), '-m') & strcmp(call(2:end), 'dimpred'), 1);
    if ~isempty(i_m)
        args = call(i_m+2:end);
        return
    end
end
if isempty(calls)
    seen = 'none';
else
    seen = strjoin(cellfun(@(c) strjoin(c', ' '), calls', 'UniformOutput', false), ' | ');
end
testCase.assertFail(sprintf('Python was never called as "python -m dimpred ...". Calls: %s', seen));
end

function value = option_value(args, option)
% Value of a command line option given as "--option value" or
% "--option=value", or '' if the option is missing
value = '';
i_option = find(strcmp(args, option), 1);
if ~isempty(i_option) && i_option < numel(args)
    value = args{i_option + 1};
    return
end
i_option = find(startsWith(args, [option '=']), 1);
if ~isempty(i_option)
    value = args{i_option}(numel(option)+2:end);
end
end

function err = verify_python_fails(testCase, f)
% f() has to give the error dimpred:pythonFailed. Returns the error.
err = error_of(f);
testCase.assertNotEmpty(err, 'dimpred_extract_features should give an error when Python fails');
testCase.verifyEqual(err.identifier, 'dimpred:pythonFailed', ...
    ['When Python fails, the error should be dimpred:pythonFailed, got: ' err.message]);
end

function set_dimpred_python(testCase, value)
% Sets the environment variable DIMPRED_PYTHON for this test only
testCase.addTeardown(@setenv, 'DIMPRED_PYTHON', getenv('DIMPRED_PYTHON'));
setenv('DIMPRED_PYTHON', value);
end

function images = make_image_files(testCase, names, folder_name)
% Empty files of the given names in a new temporary folder (optionally in
% a subfolder of the given name). Returns their full paths as a cell
% column in the order of names, repeats included.
folder = make_temp_folder(testCase);
if exist('folder_name', 'var')
    folder = fullfile(folder, folder_name);
    mkdir(folder);
end
images = fullfile(folder, names(:));
unique_images = unique(images);
for i_image = 1:numel(unique_images)
    fid = fopen(unique_images{i_image}, 'w');
    testCase.assertGreaterThan(fid, 0, sprintf('Could not create %s', unique_images{i_image}));
    fclose(fid);
end
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
