% function results = run_dimpred_tests(mode)
%
% Runs all MATLAB tests of dimpred (the test_*.m files in this folder) and
% gives an error if any test fails, so that it can be used from the command
% line or in automatic checks:
%
%   matlab -batch "cd('dimpred/tests/matlab'); run_dimpred_tests"
%
% Tests that are skipped (e.g. the feature extraction when DIMPRED_PYTHON
% is not set) do not count as failed. The reason for skipping is printed.
%
% Besides what the help of each function says, the tests expect the
% following of the MATLAB functions (the same behavior as in Python, where
% MATLAB needs its own rule):
%   Error identifiers
%     dimpred:unknownModel       dimpred_load_model: unknown model name, or
%                                a model file that does not exist (the
%                                message lists the shipped models);
%                                dimpred_list_models: the folder
%                                dimpred/models is missing
%     dimpred:inconsistentModel  dimpred_load_model: sizes in the model file
%                                do not fit together, or a variable such as
%                                target_mean is missing;
%                                dimpred_extract_features, dimpred_rise: a
%                                model struct without one of the fields
%     dimpred:folderNotFound     dimpred_find_images: the folder does not
%                                exist or is a file
%     dimpred:noImages           dimpred_find_images: no image files;
%                                dimpred_extract_features, dimpred_rise: an
%                                empty list of images
%     dimpred:fileNotFound       dimpred_extract_features, dimpred_rise: an
%                                image file does not exist, or a folder was
%                                given (checked before Python starts)
%     dimpred:pythonFailed       dimpred_extract_features, dimpred_rise:
%                                Python cannot be started or exits with an
%                                error (the message includes what Python
%                                printed)
%     dimpred:tooManyImages      dimpred_rise: the maps would need more than
%                                2 GB in the .mat file of Python (more than
%                                162 images with 66 dimensions), or the
%                                command line is too long for Windows
%     dimpred:wrongFeatureCount  dimpred_predict: the number of feature
%                                columns does not fit the model (also for
%                                transposed features and for a column vector)
%     dimpred:notFinite          dimpred_predict: NaN or Inf in the features;
%                                dimpred_similarity: NaN or Inf in the
%                                embedding
%     dimpred:tooFewObjects      dimpred_similarity: fewer than 3 objects
%     dimpred:unknownMethod      dimpred_similarity: method other than
%                                'spose' or 'dot'
%     dimpred:notMatrix          dimpred_similarity: input is not 2D
%   Other rules
%     - dimpred_list_models returns a sorted cell column
%     - dimpred_find_images returns a cell column of absolute paths, also
%       for a relative folder
%     - dimpred_load_model finds the shipped models from any current folder
%       and gives model.file as an absolute path
%     - dimpred_extract_features returns single features (images x
%       features) and, as second output, the given files as a cell column;
%       it calls python -m dimpred <images> --features-only with --model,
%       --batch-size and --device; Python is cfg.python, else
%       DIMPRED_PYTHON, else python3; many images are split into several
%       Python runs by the length of the command line in bytes
%     - dimpred_extract_features and dimpred_rise give Python a model
%       struct as it is (saved to a temporary model file)
%     - dimpred_predict never transposes the features and returns double
%     - dimpred_rise calls python -m dimpred <images> --rise with --n-masks,
%       --model, --batch-size, --out (a .mat file) and, if given, --png and
%       --device, and returns the fields relevance, dimension_maps,
%       embedding, labels, files, view, model and settings (images first)
%
% Input:
%   mode: 'all' (default) runs all tests
%         'fast' leaves out test_dimpred_extract_features and
%         test_dimpred_rise, which start Python and load the networks (a
%         few minutes). The tests in test_dimpred_extract_features_errors
%         and test_dimpred_rise_errors need no network and stay.
%
% Output:
%   results: matlab.unittest.TestResult of all tests (only returned if
%            asked for)
%
% Example:
%   run_dimpred_tests          % all tests
%   run_dimpred_tests('fast')  % without feature extraction
%   setenv('DIMPRED_PYTHON', '/path/to/python'); run_dimpred_tests  % with extraction
%
% Hebartlab, 2026/09/30
%
% See also RUNTESTS, TEST_DIMPRED_PREDICT, TEST_DIMPRED_EXTRACT_FEATURES

% History:
% 2026/10/02: more error identifiers in the list (notFinite,
%   tooManyImages, noImages and inconsistentModel of more functions)
% 2026/10/02: 'fast' also leaves out test_dimpred_rise
% 2026/09/30: the fast error tests of the extraction are now in their own
%   file, so that 'fast' keeps them; 'fast' gives an error if it finds no
%   slow tests to leave out; list of the rules the tests expect
% 2026/09/30: written together with the first MATLAB tests

function results = run_dimpred_tests(mode)

if ~exist('mode', 'var') || isempty(mode), mode = 'all'; end

% The tests are in the folder of this file, the dimpred functions in
% ../../matlab. The tests add that folder to the path themselves, we add it
% here as well so that the path is right for everything that runs below.
here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
old_path = addpath(fullfile(repo, 'matlab'), here);
restore_path = onCleanup(@() path(old_path)); % restores the path when we leave

% Collect the tests
suite = matlab.unittest.TestSuite.fromFolder(here);
switch lower(mode)
    case 'all'
        % keep all
    case 'fast'
        slow_files = {'test_dimpred_extract_features/', 'test_dimpred_rise/'};
        is_slow = false(size(suite));
        for i_file = 1:numel(slow_files)
            in_file = startsWith({suite.Name}, slow_files{i_file});
            if ~any(in_file)
                % otherwise a renamed file would silently make 'fast' run everything
                error('dimpred:noSlowTests', ['Found no tests of %s to leave out. If the file was renamed, ' ...
                    'update the name in run_dimpred_tests.'], slow_files{i_file}(1:end-1));
            end
            is_slow = is_slow | in_file;
        end
        suite = suite(~is_slow);
    otherwise
        error('dimpred:unknownMode', 'Unknown mode ''%s'' for run_dimpred_tests. Use ''all'' or ''fast''.', mode)
end

% Run them
runner = matlab.unittest.TestRunner.withTextOutput;
res = runner.run(suite);

% Summary
n_failed = nnz([res.Failed]);
n_skipped = nnz([res.Incomplete] & ~[res.Failed]);
n_passed = nnz([res.Passed]);
fprintf('\ndimpred MATLAB tests: %i passed, %i failed, %i skipped (of %i)\n', n_passed, n_failed, n_skipped, numel(res));
if n_failed > 0
    failed_names = {res([res.Failed]).Name};
    fprintf('Failed:\n');
    fprintf('  %s\n', failed_names{:});
    error('dimpred:testsFailed', '%i of %i dimpred tests failed, see above.', n_failed, numel(res))
end

if nargout > 0
    results = res;
end
