using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace TimeSeriesForecast
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<WeatherData>(@"C:\Users\Asmus\source\repos\TimeSeriesForecast\TestDataFile.csv",
                hasHeader: true, separatorChar: ',');

            IDataView dataView = context.Data.LoadFromTextFile<WeatherData>(@"C:\Users\Asmus\source\repos\TimeSeriesForecast\TestDataFile.csv", hasHeader: true, separatorChar: ',');

            var pipeline = context.Forecasting.ForecastBySsa(
                nameof(WeatherForecast.Forecast),
                nameof(WeatherData.Temp),
                windowSize: 7,
                seriesLength: 30,
                trainSize: 365,
                horizon: 7);

            var model = pipeline.Fit(data);

            var forecastingEngine = model.CreateTimeSeriesEngine<WeatherData, WeatherForecast>(context);

            var forecasts = forecastingEngine.Predict();

            Evaluate(data, model, context);

            foreach (var forecast in forecasts.Forecast)
            {
                Console.WriteLine(forecast);
            }

            Console.ReadLine();
        }

        static void Evaluate(IDataView testData, ITransformer model, MLContext mLContext)
        {
            IDataView predictions = model.Transform(testData);
            IEnumerable<float> actual = mLContext.Data.CreateEnumerable<WeatherData>(testData, true).Select(observed => observed.Temp);
            IEnumerable<float> forecast = mLContext.Data.CreateEnumerable<WeatherForecast>(predictions, true).Select(prediction => prediction.Forecast[0]);

            var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);
            var MAE = metrics.Average(error => Math.Abs(error));
            var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2)));
            
            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine("---------------------");
            Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
            Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
        }
    }
}