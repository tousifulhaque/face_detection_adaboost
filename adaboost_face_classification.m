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
%%

% prediction
result_image_grey = read_gray('data/test_face_photos/DSC01181.JPG');
% result_image_grey = read_gray('data/test_face_photos/clintonAD2505_468x448.JPG');
[color, boxes] = boosted_detector_demo(result_image_grey, 0.25:0.25:1,best_classifiers, weak_classifiers, [100, 100], 2);
imshow(color, []);

%%
%labeling test data
test_face = load('test_faces.mat').face_images;
test_non_face = load('test_nonface.mat').non_face_test;
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

%%testing 
responses = zeros(size(test_data, 3),1);
for i = 1:size(test_data,3)
    image = test_data(: , :, i);

    [max_response , ~] = boosted_multiscale_search(image, 1, best_classifiers, weak_classifiers, [100, 100]);
    responses(i) = sum(max_response(:));
    
end

prediction = responses  > 0;
face_length = size(test_face, 3);
non_face_length = size(test_non_face, 3);
false_negative = sum(prediction(1:face_length) == 0);
false_positive = sum(prediction(face_length+1 : end)==1);
accuracy = (total_test_size - false_positive -false_negative) /total_test_size;
disp(accuracy);

%%
s = filesep;
pattern = ['data', s , 'test_face_photos', s, '*.jpg'];
face_photos = dir(pattern);
% for photo_no = 1:size(face_photos)
for photo_no = 6
    face_photo_path = [face_photos(photo_no).folder ,s, face_photos(photo_no).name];
    color_face_photo = imread(face_photo_path);
    grey_face_photo = read_gray(face_photo_path);
%     imshow(grey_face_photo,[]);
    result_number = 10;
    [result, boxes] = boosted_detector_demo(grey_face_photo, .5:.2:2, best_classifiers, ...
                         weak_classifiers, [face_vertical, face_horizontal], result_number);

    
%     for number = 1:result_number
%         grey_face_photo = draw_rectangle1(grey_face_photo, boxes(number, 1), boxes(number, 2), ...
%                                  boxes(number, 3), boxes(number, 4));
%     end
    negative_histogram = read_double_image('negatives.bin');
    positive_histogram = read_double_image('positives.bin');
    
    % Apply skin detection
    box_accepted = zeros(result_number);
    for box_no = 1:result_number
        coord = boxes(box_no, 1:4);
        if any(coord(:)<1)
            continue
        end
        disp(coord);
        boxes(box_no, 2) = min(boxes(box_no, 2) , size(color_face_photo,1));
        boxes(box_no, 4) = min(boxes(box_no, 4) , size(color_face_photo,2));

        image = imresize(color_face_photo(boxes(box_no, 1): boxes(box_no, 2), ...
                            boxes(box_no, 3): boxes(box_no, 4), :), [100,100]);
        size(image);
        skin_detection = detect_skin(image, positive_histogram,  negative_histogram);
        skin_threshold = 0.6; % threshold to binarize skin detection image
        skin_pixels = skin_detection > skin_threshold;
        test_sum = sum(sum(skin_pixels(skin_pixels==1)))
        if sum(sum(skin_pixels(skin_pixels==1))) / (size(skin_pixels,1)*size(skin_pixels,2)) > 0.09
            box_accepted(box_no) = 1;
            grey_face_photo = draw_rectangle1(grey_face_photo, boxes(box_no, 1), boxes(box_no, 2), ...
                                    boxes(box_no, 3), boxes(box_no, 4));
        end
%         figure; imshow(skin_pixels, []);
    end
    figure(photo_no);imshow(grey_face_photo,[]);
end



