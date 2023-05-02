function cropping_image(crop_ratio, non_face_path, face_path, mat_file_name)

% getting the path of all test files
s = filesep;
non_face_pattern = [non_face_path, s, '*.jpg'];
face_pattern = [face_path , s, '*.bmp'];
face_list = dir(face_pattern);
non_face_list = dir(non_face_pattern);

%reading one train image to get size
file_path = [face_list(1).folder ,s, face_list(1).name];
img = read_gray(file_path);
train_face_size = size(img,1);

%taking a random window and cropping it
number_of_random_points = 50;
nonface_images = zeros(round(train_face_size*crop_ratio),round(train_face_size*crop_ratio),size(non_face_list,1) * number_of_random_points);
non_face_image_index = 1;
for i = 1 : size(non_face_list,1)
    file_path = [non_face_list(i).folder ,s, non_face_list(i).name];
    img = read_gray(file_path);
    row_range = size(img,1) - train_face_size;
    col_range = size(img,2) - train_face_size;

    row_points = randi([1 row_range], number_of_random_points);
    col_points = randi([1 col_range], number_of_random_points);
    for j = 1 : number_of_random_points
        nonface_images(:,:,non_face_image_index) = img(row_points(j):row_points(j)+99,col_points(j):col_points(j)+99);
        non_face_image_index = non_face_image_index + 1;
    end
end

save(mat_file_name, 'nonface_images');

end