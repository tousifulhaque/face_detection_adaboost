function score = adaboost_face_classification()
%%
face = load('training_face.mat').face_images;
non_face = load('training_nonface.mat').non_face_training;

face_vertical = size(face,1);
face_horizontal = size(face, 2);

number = 1000;
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
best_classifiers = AdaBoost(responses, labels, 15);












