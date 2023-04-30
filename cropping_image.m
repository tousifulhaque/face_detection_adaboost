
function non_face_training = cropping_image()
% non face croping
% non face croping

%%
face_training = dir('data/training_faces/*.bmp');
non_face_training_list = dir('data/training_nonfaces/*.jpg');
crop_ratio = 1;

s = filesep;
% size_list = size(face_training,1);
file_path = [face_training(1).folder ,s, face_training(1).name];
img = read_gray(file_path);
train_face_size = size(img,1);

number_of_random_points = 50;
non_face_training = zeros(round(train_face_size*crop_ratio),round(train_face_size*crop_ratio),size(non_face_training_list,1) * number_of_random_points);
non_face_image_index = 1;
for i = 1 : size(non_face_training_list,1)
    file_path = [non_face_training_list(i).folder ,s, non_face_training_list(i).name];
    img = read_gray(file_path);
    row_range = size(img,1) - train_face_size;
    col_range = size(img,2) - train_face_size;
%     disp([row_range col_range]);
    row_points = randi([1 row_range], number_of_random_points);
    col_points = randi([1 col_range], number_of_random_points);
    for j = 1 : number_of_random_points
        non_face_training(:,:,non_face_image_index) = img(row_points(j):row_points(j)+99,col_points(j):col_points(j)+99);
        non_face_image_index = non_face_image_index + 1;
    end
end
save('training_nonface.mat', "non_face_training");
%%
end