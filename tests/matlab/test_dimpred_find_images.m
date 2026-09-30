% function tests = test_dimpred_find_images
%
% Tests for dimpred_find_images, which lists the image files in a folder.
%
% It should return all files directly in the folder (not in subfolders)
% with the extensions .jpg .jpeg .png .bmp .tif .tiff .webp in any upper
% or lower case, ignore hidden files (starting with '.'), sort by file name
% (plain character order, as Python's sorted()) and return absolute paths
% as a cell column, also for a folder given relative to the current
% folder. A folder that does not exist or has no images gives an error.
% Each test makes its own temporary folder with empty files, which is
% deleted afterwards.
%
% Most tests compare the file names only. That the paths are absolute and
% point to the files is tested separately, so that we do not depend on
% how the folder part is spelled.
%
% The sorting is only meant for listing a folder. It is not the THINGS
% order of the reference images (see training/build_models.py), which is
% why dimpred_extract_features keeps the order of the files it is given.
%
% The Python tests in tests/ check the same behavior.
%
% Run with
%   runtests('test_dimpred_find_images')
% or all MATLAB tests with run_dimpred_tests.
%
% Martin Hebart, 2026/09/30
%
% See also DIMPRED_FIND_IMAGES, RUN_DIMPRED_TESTS

% History:
% 2026/09/30: after the second review: sorting also if dir does not sort
% 2026/09/30: after review: compare file names and test absolute paths
%   separately, relative folder, failure messages for all checks
% 2026/09/30: written before the code (test-driven development)

function tests = test_dimpred_find_images
tests = functiontests(localfunctions);
end


%% Setup

function setupOnce(testCase)

% Add the dimpred MATLAB functions to the path (restored in teardownOnce)
here = fileparts(mfilename('fullpath')); % .../dimpred/tests/matlab
repo = fileparts(fileparts(here));
testCase.TestData.old_path = addpath(fullfile(repo, 'matlab'));

end

function teardownOnce(testCase)
path(testCase.TestData.old_path);
end


%% Which files are found

function test_finds_image_extensions_in_any_case(testCase)
% macOS usually has a file system that ignores case, so there
% dir('*.jpg') also finds b.JPG. An implementation that searches one
% pattern per extension then finds all files on the Mac, but misses the
% upper case files on Linux. On the Mac this test still catches files that
% are listed twice (found with *.jpg and with *.JPG), the missing files
% only show up on Linux.
images = {'a.jpg'; 'b.JPG'; 'c.jpeg'; 'd.Jpeg'; 'e.png'; 'f.PNG'; 'g.bmp'; 'h.tif'; 'i.TIFF'; 'j.webp'};
others = {'notes.txt'; 'data.mat'; 'k.gif'; 'jpg'; 'archive.jpg.zip'; 'no_extension'};
folder = folder_with_files(testCase, [images; others]);
testCase.verifyEqual(found_names(folder), images, ...
    'Only files with image extensions (any case) should be found, each once');
end

function test_ignores_hidden_files(testCase)
% e.g. the ._name.jpg files that macOS writes on some drives
folder = folder_with_files(testCase, {'a.jpg'; '.hidden.jpg'; '._a.jpg'; 'b.png'});
testCase.verifyEqual(found_names(folder), {'a.jpg'; 'b.png'}, ...
    'Files starting with a dot should be ignored');
end

function test_does_not_search_subfolders(testCase)
folder = folder_with_files(testCase, {'a.jpg'});
mkdir(fullfile(folder, 'sub'));
make_empty_file(fullfile(folder, 'sub', 'b.jpg'));
mkdir(fullfile(folder, 'folder.jpg')); % a folder with an image extension is not an image
testCase.verifyEqual(found_names(folder), {'a.jpg'}, ...
    'Only image files directly in the folder should be found');
end


%% Order and format

function test_sorted_by_file_name(testCase)
% Plain character order: upper case before lower case, digits and '_' by
% their character code (1 < 2 < _), no natural sorting (a10 before a2)
names = {'camera_lens.jpg'; 'camera2.jpg'; 'apple.jpg'; 'a2.png'; 'camera1.jpg'; 'Zebra.jpg'; 'a10.png'};
expected = {'Zebra.jpg'; 'a10.png'; 'a2.png'; 'apple.jpg'; 'camera1.jpg'; 'camera2.jpg'; 'camera_lens.jpg'};
folder = folder_with_files(testCase, names);
testCase.verifyEqual(found_names(folder), expected, ...
    'Files should be sorted by file name in plain character order');
end

function test_sorted_also_if_dir_does_not_sort(testCase)
% On macOS and Linux, dir already returns the names in character order,
% so the test above would also pass without the sort in
% dimpred_find_images. On Windows, dir gives another order (upper and
% lower case mixed). To test the sort everywhere, we put a dir.m on the
% path that returns the entries in reverse order, only for the call of
% dimpred_find_images.
names = {'camera_lens.jpg'; 'camera2.jpg'; 'apple.jpg'; 'a2.png'; 'camera1.jpg'; 'Zebra.jpg'; 'a10.png'};
expected = {'Zebra.jpg'; 'a10.png'; 'a2.png'; 'apple.jpg'; 'camera1.jpg'; 'camera2.jpg'; 'camera_lens.jpg'};
folder = folder_with_files(testCase, names);
shadow_folder = make_temp_folder(testCase);
fid = fopen(fullfile(shadow_folder, 'dir.m'), 'w');
testCase.assertGreaterThan(fid, 0, sprintf('Could not create %s', fullfile(shadow_folder, 'dir.m')));
fprintf(fid, 'function entries = dir(varargin)\n');
fprintf(fid, 'entries = builtin(''dir'', varargin{:});\n');
fprintf(fid, 'entries = entries(end:-1:1);\n');
fclose(fid);
old_warning = warning('off', 'MATLAB:dispatcher:nameConflict'); % our dir.m shadows the dir of MATLAB
testCase.addTeardown(@warning, old_warning);
old_path = addpath(shadow_folder);
restore_path = onCleanup(@() path(old_path)); % also after an error
found = found_names(folder);
clear restore_path % the dir of MATLAB again from here on
testCase.verifyEqual(found, expected, ...
    'Files should be sorted by file name in plain character order, also if dir returns another order');
end

function test_returns_absolute_paths_as_cell_column(testCase)
folder = folder_with_files(testCase, {'b.jpg'; 'a.jpg'});
files = dimpred_find_images(folder);
testCase.verifyClass(files, 'cell', 'dimpred_find_images should return a cell array');
testCase.assertSize(files, [2 1], 'Files should be returned as a cell column');
verify_absolute_paths_in(testCase, files, folder);
testCase.verifyEqual(file_names(files), {'a.jpg'; 'b.jpg'}, 'Files should be sorted by name');
end

function test_relative_folder_gives_absolute_paths(testCase)
% A folder given relative to the current folder gives absolute paths, so
% that the list still works after the current folder has changed
parent = make_temp_folder(testCase);
folder = fullfile(parent, 'my images');
mkdir(folder);
make_empty_file(fullfile(folder, 'a.jpg'));
change_folder(testCase, parent);
files = dimpred_find_images('my images');
testCase.assertSize(files, [1 1], 'The folder ''my images'' contains one image');
verify_absolute_paths_in(testCase, files, folder);
end


%% Errors

function test_missing_folder_gives_error(testCase)
testCase.verifyError(@() dimpred_find_images(fullfile(tempname, 'no_such_folder')), 'dimpred:folderNotFound', ...
    'A folder that does not exist should give an error');
end

function test_file_instead_of_folder_gives_error(testCase)
folder = folder_with_files(testCase, {'a.jpg'});
testCase.verifyError(@() dimpred_find_images(fullfile(folder, 'a.jpg')), 'dimpred:folderNotFound', ...
    'An image file instead of a folder should give an error');
end

function test_folder_without_images_gives_error(testCase)
folder = folder_with_files(testCase, {'notes.txt'; '.hidden.jpg'});
testCase.verifyError(@() dimpred_find_images(folder), 'dimpred:noImages', ...
    'A folder without (visible) images should give an error');
end


%% Helpers

function names = found_names(folder)
% Names (without folder) of the files that dimpred_find_images finds
names = file_names(dimpred_find_images(folder));
end

function names = file_names(files)
% File names without folder, as a cell column
[~, base, ext] = cellfun(@fileparts, files(:), 'UniformOutput', false);
names = strcat(base, ext);
end

function verify_absolute_paths_in(testCase, files, folder)
% Each file is an absolute path of an existing file in folder
for i_file = 1:numel(files)
    fname = files{i_file};
    testCase.verifyTrue(is_absolute_path(fname), sprintf('%s should be an absolute path', fname));
    is_file = exist(fname, 'file') == 2;
    testCase.verifyTrue(is_file, sprintf('%s should be an existing file', fname));
    if is_file
        testCase.verifyEqual(resolve_folder(fileparts(fname)), resolve_folder(folder), ...
            sprintf('%s should be in the folder %s', fname, folder));
    end
end
end

function folder = folder_with_files(testCase, names)
% Temporary folder with empty files of the given names, deleted after the test
folder = make_temp_folder(testCase);
for i_file = 1:numel(names)
    make_empty_file(fullfile(folder, names{i_file}));
end
end

function folder = make_temp_folder(testCase)
% Temporary folder that is deleted after the test
folder = tempname;
mkdir(folder);
testCase.addTeardown(@rmdir, folder, 's');
end

function change_folder(testCase, folder)
% cd to folder for this test only. Teardowns run in reverse order, so we
% are back in the old folder before the temporary folder is deleted.
old_folder = cd(folder);
testCase.addTeardown(@cd, old_folder);
end

function make_empty_file(fname)
fid = fopen(fname, 'w');
assert(fid > 0, 'Could not create %s', fname);
fclose(fid);
end

function folder = resolve_folder(folder)
% Full path of a folder without '..' and symbolic links, so that two
% spellings of the same folder compare equal (on macOS, /var is a link to
% /private/var)
old_folder = cd(folder);
folder = pwd;
cd(old_folder);
end

function out = is_absolute_path(fname)
out = startsWith(fname, '/') || ~isempty(regexp(fname, '^[A-Za-z]:[\\/]', 'once'));
end
