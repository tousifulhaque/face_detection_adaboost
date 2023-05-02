%%making test data
s = filesep;
crop_ratio = 1;
face_path = ['data' ,s,'test_cropped_faces'];
test_face_file = 'test_faces';
make_face_mat(face_path, test_face_file, crop_ratio);

%making test nonface images
nonface_path = ['data' ,s,'test_nonfaces'];
test_nonface_file = 'test_nonfaces';
cropping_image(crop_ratio, nonface_path, face_path, test_nonface_file);


%labeling test data
test_face = load('test_faces.mat').face_images;
test_non_face = load('test_nonfaces.mat').nonface_images;
weak_classifiers = load('weak_classifiers.mat').weak_classifiers;
best_classifiers = load('best_classifiers.mat').best_classifiers;
[face_horizontal, face_vertical, ~] = size(test_face);
test_data = cat(3, test_face, test_non_face);
total_test_size = size(test_data,3);
test_labels = zeros(size(test_data, 3),1);
for i = 1 : total_test_size
    if i < size(test_face,3)+1
        test_labels(i) = 1;
    else
        test_labels(i) = 0;
    end
end

%testing 
for round = 1:size(best_classifiers,1)
    responses = zeros(size(test_data, 3),1);
    for i = 1:size(test_data,3)
        image = test_data(: , :, i);
    
        [max_response , ~] = boosted_multiscale_search(image, 1, best_classifiers(1:round,:), weak_classifiers, [face_horizontal, face_vertical]);
        responses(i) = sum(max_response(:));
        
    end
    
    prediction = responses  > 0;
    face_length = size(test_face, 3);
    non_face_length = size(test_non_face, 3);
    false_negative = sum(prediction(1:face_length) == 0);
    false_positive = sum(prediction(face_length+1 : end)==1);
    accuracy = (total_test_size - false_positive -false_negative) /total_test_size;
    disp([round false_negative, false_positive, accuracy]);
end
