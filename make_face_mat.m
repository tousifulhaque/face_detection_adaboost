function face_images = make_face_mat(path, mat_file_name)
crop_ratio  = 1;
s = filesep;
pattern = [path , s , '*.bmp'];
face_training = dir(pattern);
test_face = imread([face_training(1).folder ,s, face_training(1).name]);
total_face = size(face_training,1);
img_shape = size(test_face);
face_images = zeros(round(img_shape(1)*crop_ratio), round(img_shape(2)*crop_ratio), total_face);
upper = ceil((img_shape(1) * crop_ratio) /2);
lower = floor((img_shape(1) * crop_ratio) /2); 
row_center = img_shape(1) / 2;
col_center = img_shape(2) / 2;

for i = 1:total_face
     img = imread([face_training(i).folder ,s, face_training(i).name]);
     face_images(:,:,i) = img(row_center-upper+1 : row_center+lower, col_center - upper+1 : col_center + lower); 
end

save(mat_file_name, 'face_images');
end



