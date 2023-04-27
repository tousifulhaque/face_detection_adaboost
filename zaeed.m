s = filesep;
repo_path = 'C:\Files\Image processing 2023\face_detection_adaboost';
addpath([repo_path s  'boosting'])
addpath([repo_path s 'data'])
cd([repo_path s])
%% 

% non face croping
face_training = dir('data/training_faces/*.bmp');
non_face_training_list = dir('data/training_nonfaces/*.jpg');


s = filesep;
% size_list = size(face_training,1);
file_path = [face_training(1).folder ,s, face_training(1).name];
img = imread(file_path);
train_face_size = size(img,1);

number_of_random_points = 50;
non_face_training = zeros(train_face_size,train_face_size,3,size(non_face_training_list,1) * number_of_random_points);
non_face_image_index = 1;
for i = 1 : size(non_face_training_list,1)
    file_path = [non_face_training_list(i).folder ,s, non_face_training_list(i).name];
    img = imread(file_path);
    row_range = size(img,1) - train_face_size;
    col_range = size(img,2) - train_face_size;
%     disp([row_range col_range]);
    row_points = randi([1 row_range], number_of_random_points);
    col_points = randi([1 col_range], number_of_random_points);
    for j = 1 : number_of_random_points
        non_face_training(:,:,:,non_face_image_index) = img(row_points(j):row_points(j)+99,col_points(j):col_points(j)+99,:);
        non_face_image_index = non_face_image_index + 1;
    end
end
% disp(R);
figure(1);
% imshow(non_face_training(:,:,:,2));
disp(non_face_training(1:5,1:5,:,2));
%%