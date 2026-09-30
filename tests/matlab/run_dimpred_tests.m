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
%                                message lists the shipped models)
%     dimpred:inconsistentModel  dimpred_load_model: sizes in the model file
%                                do not fit together, or a variable such as
%                                target_mean is missing
%     dimpred:folderNotFound     dimpred_find_images: the folder does not
%                                exist or is a file
%     dimpred:noImages           dimpred_find_images: no image files
%     dimpred:fileNotFound       dimpred_extract_features: an image file does
%                                not exist, or a folder was given (checked
%                                before Python starts)
%     dimpred:pythonFailed       dimpred_extract_features: Python cannot be
%                                started or exits with an error (the message
%                                includes what Python printed)
%     dimpred:wrongFeatureCount  dimpred_predict: the number of feature
%                                columns does not fit the model (also for
%                                transposed features and for a column vector)
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
%       DIMPRED_PYTHON, else python3
%     - dimpred_predict never transposes the features and returns double
%
% Input:
%   mode: 'all' (default) runs all tests
%         'fast' leaves out test_dimpred_extract_features, which starts
%         Python and loads the networks (a few minutes). The tests in
%         test_dimpred_extract_features_errors need no network and stay.
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
% Martin Hebart, 2026/09/30
%
% See also RUNTESTS, TEST_DIMPRED_PREDICT, TEST_DIMPRED_EXTRACT_FEATURES

% History:
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
        is_slow = startsWith({suite.Name}, 'test_dimpred_extract_features/');
        if ~any(is_slow)
            % otherwise a renamed file would silently make 'fast' run everything
            error('dimpred:noSlowTests', ['Found no tests of test_dimpred_extract_features to leave out. ' ...
                'If the file was renamed, update the name in run_dimpred_tests.']);
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
