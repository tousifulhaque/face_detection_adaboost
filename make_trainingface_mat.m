function face_images = make_trainingface_mat(path)

s = filesep;
pattern = [path , s , '*.bmp'];
face_training = dir(pattern);
test_face = imread([face_training(1).folder ,s, face_training(1).name]);
total_face = size(face_training,1);
img_shape = size(test_face);
face_images = zeros(img_shape(1), img_shape(2), total_face);

for i = 1:total_face
     face_images(: ,:, i) = imread([face_training(i).folder ,s, face_training(i).name]);
end

save('training_face.mat', 'face_images');
end



