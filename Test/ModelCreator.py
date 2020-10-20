class modelCreator:
    def __init__(self, model_name, base_model, Dense_Layer_Configurations, AddDropOut=True, Dropout_Float=0.2,
                 Freeze_Model_Layer=True):
        self.model_name = model_name
        self.base_model = base_model
        self.Dense_Layer_Configurations = Dense_Layer_Configurations
        self.AddDropOut = AddDropOut
        self.Dropout_Float = Dropout_Float
        self.Freeze_Model_Layer = Freeze_Model_Layer

    def configureModel(self, lastLayer):
        previousLayer = lastLayer
        index = 0

        for neurons in self.Dense_Layer_Configurations[:-1]:  # Do not include the last layer as it is definitely softmax.  Adding it seperately.
            fc = Dense(neurons, activation='relu')(previousLayer)
            index += 1
            previousLayer = fc
            if (self.AddDropOut):
                fc = Dropout(self.Dropout_Float)(previousLayer)
                previousLayer = fc
                index += 1

        # Adding softmax
        softmaxLayer = Dense(3, activation='softmax')(previousLayer)
        index += 1
        self.model = Model(self.base_model.inputs, softmaxLayer)

        if (
        self.Freeze_Model_Layer):  # This is to determine whether we should only train weights that are in the pre-trained network.
            for layer in self.model.layers[:-1 * index]:  # Only go up to last flatten layer.
                layer.trainable = False

    def combineModels(self):
        # Freeze models
        if (self.model_name.lower() == "vgg19"):

            baseOutput = self.base_model.get_layer("block5_pool").output  # Block5_pool is the last output
            flatten = Flatten()(baseOutput)  # Add the flatten layer

            self.configureModel(flatten)

            #             for neurons in self.Dense_Layer_Configurations[:-1]:
            #                 fc=Dense(neurons, activation='relu')(previousLayer)
            #                 index+=1
            #                 if(self.AddDropOut):
            #                     fc=Dropout(self.Dropout_Float)(fc)
            #                     index+=1
            #                 previousLayer=fc

            #             softmaxLayer=Dense(3, activation='softmax')(previousLayer)
            #             index+=1
            #             self.model=Model(self.base_model.inputs,softmaxLayer)

            #             if(self.Freeze_Model_Layer):
            #                 for layer in self.model.layers[:-1*index]:
            #                     layer.trainable=False
            return self.model

        elif (self.model_name.lower() == "resnet"):  # too bad python doesn't have switch statements....
            baseOutput = self.base_model.get_layer("activation_294").output
            self.configureModel(baseOutput)
            return self.model
        else:
            print("Please double check your model name and try again")