% function tests = test_dimpred_rise_errors
%
% Tests for dimpred_rise that need no network and run in a few seconds:
% the help text, wrong input, a Python that fails, and the command line
% that is sent to Python. The tests that really compute heatmaps are in
% test_dimpred_rise.m.
%
% dimpred_rise runs the command line tool of the Python version
% (python -m dimpred <images> --rise ...) and loads the .mat file that it
% writes. As in test_dimpred_extract_features_errors.m, we use a fake
% "python": a small shell script that writes the arguments it gets to a
% text file, prints an error text and exits with status 1. dimpred_rise
% then has to give the error dimpred:pythonFailed, and we can read which
% arguments arrived after "-m dimpred". A shell script does not run on
% Windows, so the tests with the fake Python are skipped there. The image
% files here are empty, because Python never reads them.
%
% Run with
%   runtests('test_dimpred_rise_errors')
% or all MATLAB tests with run_dimpred_tests (also in 'fast' mode).
%
% Hebartlab, 2026/10/02
%
% See also DIMPRED_RISE, TEST_DIMPRED_RISE, RUN_DIMPRED_TESTS

% History:
% 2026/10/02: written before dimpred_rise.m (test-driven)

function tests = test_dimpred_rise_errors
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


%% Help text

function test_help_says_how_long_it_takes(testCase)
% RISE passes thousands of masked images through the network, so the help
% has to say how long it takes and that fewer masks are possible
text = evalc('help dimpred_rise');
testCase.verifySubstring(text, '10 min', 'The help should say that RN50x64 takes about 10 min per image');
testCase.verifySubstring(text, '1 min', 'The help should say that AligNet takes less than 1 min per image');
testCase.verifySubstring(text, '2000', 'The help should say that fewer masks (e.g. 2000) are possible');
end

function test_help_names_the_recommended_model_and_the_outputs(testCase)
text = evalc('help dimpred_rise');
testCase.verifySubstring(text, 'rn50x64_66d_ridge', 'The help should recommend rn50x64_66d_ridge for heatmaps');
fields = {'relevance', 'dimension_maps', 'embedding', 'labels', 'files', 'view', 'settings', 'cfg.n_masks', 'cfg.png'};
for i_field = 1:numel(fields)
    testCase.verifySubstring(text, fields{i_field}, sprintf('The help should explain %s', fields{i_field}));
end
end


%% Wrong input (checked before Python starts)

function test_missing_image_gives_error(testCase)
images = make_image_files(testCase, {'a.jpg'});
images{2, 1} = fullfile(tempname, 'missing.jpg');
cfg.python = testCase.TestData.no_python;
err = error_of(@() dimpred_rise(images, [], cfg));
testCase.assertNotEmpty(err, 'A missing image file should give an error');
testCase.verifyEqual(err.identifier, 'dimpred:fileNotFound', ...
    ['A missing image file should give dimpred:fileNotFound before Python is started, got: ' err.message]);
testCase.verifySubstring(err.message, 'missing.jpg', 'The error message should name the missing file');
end

function test_folder_gives_error(testCase)
images = make_image_files(testCase, {'a.jpg'});
folder = fileparts(images{1});
cfg.python = testCase.TestData.no_python;
err = error_of(@() dimpred_rise(folder, [], cfg));
testCase.assertNotEmpty(err, 'A folder instead of image files should give an error');
testCase.verifyEqual(err.identifier, 'dimpred:fileNotFound', ...
    ['A folder instead of image files should give dimpred:fileNotFound, got: ' err.message]);
testCase.verifySubstring(err.message, 'dimpred_find_images', ...
    'For a folder, the error message should point to dimpred_find_images');
end

function test_no_images_give_error(testCase)
cfg.python = testCase.TestData.no_python;
err = error_of(@() dimpred_rise({}, [], cfg));
testCase.assertNotEmpty(err, 'No images should give an error');
testCase.verifyEqual(err.identifier, 'dimpred:noImages', ['No images should give dimpred:noImages, got: ' err.message]);
end

function test_unknown_model_gives_error(testCase)
images = make_image_files(testCase, {'a.jpg'});
cfg.python = testCase.TestData.no_python;
err = error_of(@() dimpred_rise(images, 'no_such_model', cfg));
testCase.assertNotEmpty(err, 'An unknown model should give an error');
testCase.verifyEqual(err.identifier, 'dimpred:unknownModel', ['An unknown model should give dimpred:unknownModel, got: ' err.message]);
end


%% Python fails

function test_python_that_does_not_exist_gives_error(testCase)
images = make_image_files(testCase, {'a.jpg'});
cfg.python = testCase.TestData.no_python;
verify_python_fails(testCase, @() dimpred_rise(images, [], cfg));
end

function test_python_error_text_is_shown(testCase)
% e.g. "n_masks has to be ..." from Python has to reach the user
fake = make_fake_python(testCase, 'fake');
images = make_image_files(testCase, {'a.jpg'});
cfg.python = fake.python;
err = verify_python_fails(testCase, @() dimpred_rise(images, [], cfg));
testCase.verifySubstring(err.message, fake.error_text, 'The error message should include the text that Python printed');
end

function test_python_output_is_shown_while_it_runs(testCase)
% Python prints one line per image, and an image can take minutes, so
% what Python prints has to appear in MATLAB while it runs, not only in
% the error message at the end
fake = make_fake_python(testCase, 'fake');
images = make_image_files(testCase, {'a.jpg'});
cfg.python = fake.python;
printed = evalc('try, dimpred_rise(images, [], cfg); catch, end');
testCase.verifySubstring(printed, fake.error_text, 'What Python prints should be shown in MATLAB while it runs');
end


%% Command line sent to Python (with the fake Python)

function test_images_are_passed_in_given_order(testCase)
fake = make_fake_python(testCase, 'fake');
images = make_image_files(testCase, {'c.jpg'; 'a.jpg'; 'c.jpg'; 'b.jpg'});
cfg.python = fake.python;
verify_python_fails(testCase, @() dimpred_rise(images, [], cfg));
args = arguments_after_dimpred(testCase, fake);
testCase.verifyEqual(args(ismember(args, images)), images, ...
    'The image files should be passed to Python in the given order, with repeats');
end

function test_names_with_spaces_and_quotes_are_passed_unchanged(testCase)
fake = make_fake_python(testCase, 'fake');
images = make_image_files(testCase, {'first image.jpg'; 'it''s the second.jpg'; 'third (copy) & more.jpg'}, 'my images');
cfg.python = fake.python;
verify_python_fails(testCase, @() dimpred_rise(images, [], cfg));
args = arguments_after_dimpred(testCase, fake);
for i_image = 1:numel(images)
    testCase.verifyTrue(any(strcmp(args, images{i_image})), ...
        sprintf('The file %s should arrive in Python as one argument. Arguments were:\n%s', images{i_image}, strjoin(args', '\n')));
end
end

function test_default_options_are_passed(testCase)
% Without cfg: --rise with 6000 masks, a .mat file as output, no PNG files
fake = make_fake_python(testCase, 'fake');
images = make_image_files(testCase, {'a.jpg'});
cfg.python = fake.python;
verify_python_fails(testCase, @() dimpred_rise(images, [], cfg));
args = arguments_after_dimpred(testCase, fake);
all_args = strjoin(args', ' ');
testCase.verifyTrue(any(strcmp(args, '--rise')), ['--rise is missing: ' all_args]);
testCase.verifyEqual(option_value(args, '--n-masks'), '6000', ['The default should be --n-masks 6000: ' all_args]);
testCase.verifyFalse(any(strcmp(args, '--png')), ['Without cfg.png, no PNG files should be asked for: ' all_args]);
testCase.verifyFalse(any(strcmp(args, '--device')), ['Without cfg.device, Python should choose the device: ' all_args]);
[~, ~, extension] = fileparts(option_value(args, '--out'));
testCase.verifyEqual(extension, '.mat', ['The output of Python should be a .mat file: ' all_args]);
[~, model_name] = fileparts(option_value(args, '--model'));
testCase.verifyEqual(model_name, 'alignet_siglip2b_66d_ridge', ['Without model, the default model should be used: ' all_args]);
end

function test_options_are_passed(testCase)
% The model, cfg.n_masks, cfg.png, cfg.batch_size and cfg.device have to
% reach the command line tool with its option names
fake = make_fake_python(testCase, 'fake');
images = make_image_files(testCase, {'a.jpg'});
png_folder = fullfile(make_temp_folder(testCase), 'my png files');
cfg.python = fake.python;
cfg.n_masks = 2000;
cfg.png = png_folder;
cfg.batch_size = 5;
cfg.device = 'cpu';
verify_python_fails(testCase, @() dimpred_rise(images, 'rn50x64_66d_ridge', cfg));
args = arguments_after_dimpred(testCase, fake);
all_args = strjoin(args', ' ');
testCase.verifyEqual(option_value(args, '--n-masks'), '2000', ['cfg.n_masks = 2000 should give --n-masks 2000: ' all_args]);
testCase.verifyEqual(option_value(args, '--png'), png_folder, ['cfg.png should give --png with the folder: ' all_args]);
testCase.verifyEqual(option_value(args, '--batch-size'), '5', ['cfg.batch_size = 5 should give --batch-size 5: ' all_args]);
testCase.verifyEqual(option_value(args, '--device'), 'cpu', ['cfg.device = ''cpu'' should give --device cpu: ' all_args]);
[~, model_name] = fileparts(option_value(args, '--model'));
testCase.verifyEqual(model_name, 'rn50x64_66d_ridge', ['The model should be passed with --model: ' all_args]);
end

function test_dimpred_python_is_used_without_cfg(testCase)
fake = make_fake_python(testCase, 'fake');
testCase.addTeardown(@setenv, 'DIMPRED_PYTHON', getenv('DIMPRED_PYTHON'));
setenv('DIMPRED_PYTHON', fake.python);
images = make_image_files(testCase, {'a.jpg'});
verify_python_fails(testCase, @() dimpred_rise(images));
testCase.verifyNotEmpty(recorded_calls(fake), 'Without cfg, dimpred_rise should run the Python given in DIMPRED_PYTHON');
end


%% Helpers (as in test_dimpred_extract_features_errors.m)

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
% Arguments that the fake Python got after "-m dimpred", as a cell column
calls = recorded_calls(fake);
for i_call = 1:numel(calls)
    call = calls{i_call};
    i_m = find(strcmp(call(1:end-1), '-m') & strcmp(call(2:end), 'dimpred'), 1);
    if ~isempty(i_m)
        args = call(i_m+2:end);
        return
    end
end
testCase.assertFail('Python was never called as "python -m dimpred ...".');
end

function value = option_value(args, option)
% Value of a command line option given as "--option value", or '' if the
% option is missing
value = '';
i_option = find(strcmp(args, option), 1);
if ~isempty(i_option) && i_option < numel(args)
    value = args{i_option + 1};
end
end

function err = verify_python_fails(testCase, f)
% f() has to give the error dimpred:pythonFailed. Returns the error.
err = error_of(f);
testCase.assertNotEmpty(err, 'dimpred_rise should give an error when Python fails');
testCase.verifyEqual(err.identifier, 'dimpred:pythonFailed', ...
    ['When Python fails, the error should be dimpred:pythonFailed, got: ' err.message]);
end

function images = make_image_files(testCase, names, folder_name)
% Empty files of the given names in a new temporary folder (optionally in
% a subfolder of the given name), as a cell column in the order of names
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
