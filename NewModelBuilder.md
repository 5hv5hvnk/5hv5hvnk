## Using ModelBuilder class for deploying PyMC models <br>
The new `ModelBuilder` class  allows user to use direct methods to `fit`, `predict`,`save`, `load`. A user can create any model as they want to and just inherit the `ModelBuilder` class and use the predefined methods. <br>
Let's learn more about using an example <br>
```
class LinearModel(ModelBuilder):
    _model_type = 'LinearModel'
    version = '0.1'

    def _build(self):
        # data
        x = pm.MutableData('x', self.data['input'].values)
        y_data = pm.MutableData('y_data', self.data['output'].values)

        # prior parameters
        a_loc = self.model_config['a_loc']
        a_scale = self.model_config['a_scale']
        b_loc = self.model_config['b_loc']
        b_scale = self.model_config['b_scale']
        obs_error = self.model_config['obs_error']

        # priors
        a = pm.Normal("a", a_loc, sigma=a_scale)
        b = pm.Normal("b", b_loc, sigma=b_scale)
        obs_error = pm.HalfNormal("σ_model_fmc", obs_error)

        # observed data
        y_model = pm.Normal('y_model', a + b * x, obs_error, observed=y_data)


    def _data_setter(self, data : pd.DataFrame):
        with self.model:
            pm.set_data({'x': data['input'].values})
            try: # if y values in new data
                pm.set_data({'y_data': data['output'].values})
            except: # dummies otherwise
                pm.set_data({'y_data': np.zeros(len(data))})


    @classmethod
    def create_sample_input(cls):
        x = np.linspace(start=1, stop=9, num=15)
        y = 5 * x + 3 + np.random.normal(0, 1, len(x))
        data = pd.DataFrame({'input': x, 'output': y})

        model_config = {
            'a_loc': 7,
            'a_scale': 3,
            'b_loc': 5,
            'b_scale': 3,
            'obs_error': 2,
        }

        sampler_config = {
            'draws': 1_000,
            'tune': 1_000,
            'chains': 1,
            'target_accept': 0.95,
        }

        return data, model_config, sampler_config
```
Above is an example of user created `LinearModel` which inherits the `ModelBuilder` class and overrides methods to build, set data and creating sample input. <br>
#### Let's look st an implentation to understand the real world use of the `ModelBuilder` class in a better way <br>

First we import libraries we need to deploy a model
```
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
```
Now we import the `LinearModel` present in the ModelBuilder.py file
```
from pymc.modelBuilder import LinearModel
```
Now we can create a object of LinearModel type which we can edit according to our use or use the default model as defined by the user.
Most importantly if you make _really cool model_ and want to deploy the same it will be easier for you to make a class and share the model so people can use it via the object instead of redefining the model everytime they need it. <br>

Making the object is same as making an object of a python class, we first define parameters we need to make the object like -> data, model configration and sampler comfigration. 
We can do that using the `create_sample_input()` method as described above.
```
data, model_config, sampler_config = LinearModel.create_sample_input() 
model = LinearModel(data, model_config, sampler_config)
```
After making the object of class `LinearModel` we can fit the model using the `.fit()` method
```
idata = model.fit()
```
The `.fit()` method is defined in a manner such that it returns the `idata` which we can save in another variable as per needs of user.
```
def fit(self, data : pd.DataFrame = None):
        if data is not None: 
            self.data = data

        if self.basic_RVs == []:
            print('No model found, building model...')
            self.build()

        with self:
            self.idata = pm.sample(**self.sample_config)
            self.idata.extend(pm.sample_prior_predictive())
            self.idata.extend(pm.sample_posterior_predictive(self.idata))

        self.idata.attrs['id']=self.id()
        self.idata.attrs['model_type']=self._model_type
        self.idata.attrs['version']=self.version
        self.idata.attrs['sample_conifg']=self.sample_config
        self.idata.attrs['model_config']=self.model_config
        # model,model_type,version,sample_conifg,model_config
        return self.idata
```
`fit` method takes one argument `data` on which we need to fit the model and assigns `idata.attrs` with `id`, `model_type`, `version`, `sample_conifg`, `model_config`. 
* `id` : This is an unique id given to a model based on model_config, sample_conifg, version, model_type. User can use it to check if the model matches to other model they have defined.
* `model_type` : Model type tells us what kind of model is this in this case it outputs **Linear Model** 
* `version` : In case user want to improvise on models they can keep a track of model by it's version. As the version changes the unique hash in the `id` also changes.
* `sample_conifg` : It stores values of the sampler configration set by user for this particular model.
* `model_config` : It stores values of the model configration set by user for this particular model.

As we know we only have the object of class `LinearModel` and not the model, the fit functions call the `.build()` method if the model is not built yet. <br> <br>

After fitting the model we can probably save it to share model as a file so one can use it again.
To `save` or `load` we can easily call methods for respective tasks with the following syntax 
```
path = <path>
name = <name>
save_model = True # Boolean with defualt value True
save_idata = True # Boolean with default value True
model.save(name,path,save_model,save_idata)

load_model=True # Boolean with defualt value True
laod_idata=True # Boolean with defualt value True

imported_model = LinearModel.load(name,path,load_model,laod_idata)
``` 
This saves two files at given path and name
1. `.pickle` file that stores the model
2.  `.nc` file that stores the idata
When saving or loading the model multiple times we might not need to save the model or the idata of the model so we can change the parameters accoridngly.
<br>
`predict()` method allows users to do a posterior predcit with the fitted model.
```
# Creating new data to predict on
prediction_data = pd.DataFrame({'input': [10, 20, 50]})
# Predicting only point estimate
pred_mean = imported_model.predict(prediction_data)
# Predicitng samples
pred_samples = imported_model.predict(prediction_data, point_estimate=False)
```

After using the `predict()` we can plot our data and see graphically how good our `LinearModel` is
```
plt.plot(prediction_data.input, pred_samples['y_model'].T, color='0.5', 
         alpha=.05, zorder=0)
plt.plot(prediction_data.input, pred_mean['y_model'], label='mean', zorder=1)
plt.legend()
plt.show()
```
Plots recived : 
<img src="https://i.imgur.com/FRPgbfc.png">