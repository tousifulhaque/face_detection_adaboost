s = filesep;
repo_path = 'C:\Files\Image processing 2023\face_detection_adaboost';
addpath([repo_path s  'boosting'])
addpath([repo_path s 'data'])
cd([repo_path s])
%% 

% non face croping
file_jpg = dir('data/training_faces/*.bmp');


s = filesep;
size_list = size(file_jpg,1);

for i = 1 : 2
    file_path = [file_jpg(i).folder ,s, file_jpg(i).name];
    img = imread(file_path);
    size(img)
end
%%