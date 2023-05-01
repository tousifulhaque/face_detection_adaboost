%% cascading
% making groups out of the best classifer
tic
group_1 = best_classifiers(1:7, :);
group_2 = best_classifiers(8:18, :);
group_3 = best_classifiers(19:43, :);

% taking a image window and passing it to the detectors

% need to pass the coordinates of of the window 
face_size = [100 100];
%for all the windows

image = read_gray('data/test_face_photos/DSC02950.JPG');
size(image)

face_vertical = face_size(1);
face_horizontal = face_size(2);

scales = .5:.25:2;

max_responses = ones(size(image)) * -10;
max_scales = zeros(size(image));

for scale = scales
    scaled_image = imresize(image, 1/scale, 'bilinear');
    integral = integral_image(scaled_image);
    vertical_size = size(scaled_image, 1);
    horizontal_size = size(scaled_image, 2);
    
    window_coords = zeros((vertical_size-face_vertical)*(horizontal_size -face_horizontal),2);
    for i =  1:(vertical_size-face_vertical+1)
        curr_point = (i-1) * (horizontal_size - face_horizontal + 1);
        for j = 1:(horizontal_size - face_horizontal + 1)
            window_coords(curr_point + j,1) = i;
            window_coords(curr_point + j,2) = j;
        end
    end
    
    [~ , group_1_selection] = filter_window(group_1, weak_classifiers, window_coords, integral, vertical_size, horizontal_size, face_size);
    [~ , group_2_selection] = filter_window(group_2, weak_classifiers, group_1_selection, integral, vertical_size, horizontal_size, face_size);
    [responses, group_3_selection] = filter_window(group_3, weak_classifiers, group_2_selection, integral, vertical_size, horizontal_size, face_size);
    temp_result = imresize(responses, size(image), 'bilinear');
    
    higher_maxes = (temp_result > max_responses);
    max_scales(higher_maxes) = scale;
    max_responses(higher_maxes) = temp_result(higher_maxes);
end

boxes = detection_boxes(image, zeros(face_size), max_responses, ...
                         max_scales, 5);

result = image;

for number = 1:5
    result = draw_rectangle1(result, boxes(number, 1), boxes(number, 2), ...
                             boxes(number, 3), boxes(number, 4));
end

toc
figure(1);
imshow(result,[]);



%creating a 2-D array with all the coordinates of the image
%%



%%
function [responses, result] = filter_window(boosted, weak_classifiers, windows, integral, vertical_size, horizontal_size, face_size)
face_vertical = face_size(1);
face_horizontal = face_size(2);
responses = zeros(vertical_size, horizontal_size);
classifier_number = size(boosted, 1);
% give the window to this code and 
all_windows = windows;
selected_windows = [];
selected_windows_center = [];


    for i = 1:size(windows, 1)
        response = 0 ;
        for weak_classifier = 1:classifier_number
            classifier_index = boosted(weak_classifier, 1);
            classifier_alpha = boosted(weak_classifier, 2);
            classifier_threshold = boosted(weak_classifier, 3);
            classifier = weak_classifiers{classifier_index};
            vertical = all_windows(i , 1);
            horizontal = all_windows(i, 2);
            response1 = eval_weak_classifier(classifier, integral, vertical, horizontal);
            if (response1 > classifier_threshold)
                response2 = 1;
            else
               response2 = -1;
            end
            response = classifier_alpha * response2 + response;
    
            row = vertical + round(face_vertical/2);
            col = horizontal + round(face_horizontal/2);
            responses(row, col) = responses(row, col) + response;
        end
        if response > 1
            selected_windows = [selected_windows; windows(i, :)];

        end
    end
 
result = selected_windows;
end



%%


   

