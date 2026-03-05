function [Recall,Spec,Precision,NPV,ACC,F1Score] = performanceIndexes(CM,PositiveClass)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculates Performance Indexes for a classification task. 
% It takes the as positive class the one indicated by the user.
% Parameters of the function:
% --------------------------
% Inputs:
%   CM:             Confusion Matrix (CxC, where C is the number of classes. CM(I,J) represents the count of instances of class I and whose predicted class is J
%   PositiveClass:  the index of the positive class (between 1 and C)
% Returns:
%   Recall:     Sensitivity of the classifier.
%   Spec:       Specificity of the classifier.
%   Precision:        Positive Predicted Value
%   NPV:        Negative Predicted Value
%   ACC:        Global accuracy of the classifier.
%   F1-Score:   Armonic mean among Precision and Recall 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% transforma la CM en una de dos clases 
classNum=size(CM,1);
%verdaderos positivos
TP=CM(PositiveClass,PositiveClass);
%verdaderos negativos, falsos positivos Y falsos negativos
TN=0;
FP=0;
FN=0;
for REAL=1:classNum
        for PREDICTED=1:classNum
                if (REAL~=PositiveClass)&&(PREDICTED~=PositiveClass) 
                    TN=TN+CM(REAL,PREDICTED); 
                end 
                if (REAL~=PositiveClass)&&(PREDICTED==PositiveClass) 
                    FP=FP+CM(REAL,PREDICTED); 
                end 
                if (REAL==PositiveClass)&&(PREDICTED~=PositiveClass) 
                    FN=FN+CM(REAL,PREDICTED); 
                end
        end
end

%Recall o sensibilidad = TP/(TP+FN)
if (TP+FN)==0
    fprintf('No es posible calcular el Recall al no haber ejemplos positivos\n');
    Recall=NaN;
else
    Recall= TP/(TP+FN);
end

%Especificidad = TN/(FP+TN) 
if (TN+FP)==0
    fprintf('No es posible calcular la Especificidad al no haber ejemplos negativos\n');
    Spec=NaN;
else
    Spec= TN/(TN+FP);
end
%Precision =TP/(TP+FP)
if TP+FP==0
    fprintf('No es posible calcular la Precision al no haber ejemplos clasificados en la clase positiva\n');
    Precision=NaN;
else
    Precision=TP/(TP+FP);
end

%NPV=TN/(TN+FN) 
if TN+FN==0
    fprintf('No es posible calcular la tasa el VPN al no haber ejemplos clasificados en la clase negativa\n');
    NPV=NaN;
else
    NPV=TN/(TN+FN);
end

F1Score = 2* (Precision * Recall)/(Precision + Recall);
ACC=(TP+TN)/(TP+TN+FP+FN);
