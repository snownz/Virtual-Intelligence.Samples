using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace MultiLayerPerceptron.EvenOrOdd
{
    public static class Data
    {
        private static string dataPath = @"..\..\..\DataSet\EvenOrOdd\";

        public static (float[][] inputs, float[][] desireds) Get()
        {
            var inputString = File.ReadAllLines(dataPath + "inputs.csv").ToList();
            var outputString = File.ReadAllLines(dataPath + "outputs.csv").ToList();

            var inputs = new float[inputString.Count][];
            var outputs = new float[outputString.Count][];

            for (int i = 0; i < inputString.Count; i++)
            {
                inputs[i] = inputString[i].Split(';').Select(x => float.Parse(x)).ToArray();
            }

            for (int i = 0; i < outputString.Count; i++)
            {
                outputs[i] = outputString[i].Split(';').Select(x => float.Parse(x)).ToArray();
            }

            return (inputs, outputs);
        }
    }
}
