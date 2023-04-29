function score = adaboost_face_classification()
%%
face = load('training_face.mat').face_images;
non_face = load('training_nonface.mat').non_face_training;

face_vertical = size(face,1);
face_horizontal = size(face, 2);

number = 2000;
weak_classifiers = cell(1, number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
end


% integral image
training_data = cat(3, face, non_face);
total_train_size = size(training_data,3); 
integral_train = zeros(size(training_data));
labels = zeros(size(integral_train, 3),1);

for i = 1 : total_train_size
    integral_train(:, : , i) = integral_image(training_data(:,:,i));
    if i < size(face,3)+1
        labels(i) = 1;
    else
        labels(i) = -1;
    end
end

responses =  zeros(number, total_train_size);

for example = 1:total_train_size
    integral = integral_train(:, :, example);
    for feature = 1:number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
end

% adaboost
rounds = 43;
best_classifiers = AdaBoost(responses, labels, rounds);
%% 
boosted_threshold = 0;
for classifier_number = 1:rounds
    boosted_threshold = boosted_threshold + best_classifiers(classifier_number,2) * best_classifiers(classifier_number,3);
end
%%
% result_image = imread('data\test_face_photos\clintonAD2505_468x448.JPG');
result_image_grey = read_gray('data\test_face_photos\clintonAD2505_468x448.JPG');
result_image_integral = integral_image(result_image_grey);
face_result = zeros(size(result_image_integral));
for row = 1:25:size(result_image_integral,1) - face_vertical
    for col = 1:25:size(result_image_integral,2) - face_horizontal
        result_response = 0;
        result_image_integral_2 = integral_image(result_image_grey(row:row+face_vertical,col:col+face_horizontal));
        for i = 1:rounds
            result_response = result_response + eval_weak_classifier(weak_classifiers{best_classifiers(i,1)},result_image_integral_2) * best_classifiers(i,2);
        end
%         if result_response >= boosted_threshold
%             face_result(row,col) = 1;
%         end
        face_result(row,col) = result_response;
    end
end
% [max_val,I] = max(face_result(:));
% [row col] = ind2sub(size(face_result),I);
% face_result(row,col) = -999;
% [max_val,I] = max(face_result(:));
% [row col] = ind2sub(size(face_result),I);
% new_result = draw_rectangle1(result_image_grey,row,row+100,col,col+100);
% imshow(new_result,[]);
new_result = result_image_grey;
for i = 1:16
    [max_val,I] = max(face_result(:));
    [row, col] = ind2sub(size(face_result),I);
    face_result(row,col) = -999;
    new_result = draw_rectangle1(new_result,row,row+100,col,col+100);
end
imshow(new_result,[]);















