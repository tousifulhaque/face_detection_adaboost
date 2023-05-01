function [responses, result] = filter_window(boosted, weak_classifiers, windows, integral, vertical_size, horizontal_size, face_size)
face_row = face_size(1);
face_col= face_size(2);
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
    
            row = vertical + round(face_row/2);
            col = horizontal + round(face_col/2);
            responses(row, col) = responses(row, col) + response;
        end
        if response > 1
            selected_windows = [selected_windows; windows(i, :)];

        end
    end
 
result = selected_windows;
end
