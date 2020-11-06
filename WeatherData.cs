using Microsoft.ML.Data;
using System;

namespace TimeSeriesForecast
{
    public class WeatherData
    {
        [LoadColumn(0)]
        public DateTime Date { get; set; }

        [LoadColumn(1)]
        public float Temp { get; set; }
    }
}