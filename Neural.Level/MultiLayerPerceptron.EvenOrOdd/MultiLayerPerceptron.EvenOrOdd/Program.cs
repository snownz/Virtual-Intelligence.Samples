using System;
using VI.Neural.Factory;
using VI.Neural.Node;
using VI.NumSharp;

namespace MultiLayerPerceptron.EvenOrOdd
{
    class Program
    {
        static void Main(string[] args)
        {
            var rd = new Random();
            var values = new[] { .1f, .002f };

            var watch = System.Diagnostics.Stopwatch.StartNew();

            ProcessingDevice.Device = Device.CPU;

            watch.Stop();
            Console.WriteLine($"Device Time: {watch.ElapsedMilliseconds}ms");

            var hiddens = new LayerCreator(2, 4)
                .WithLearningRate(values[0])
                .WithMomentum(0)
                .FullSynapse()
                .Supervised()
                .DenseLayer()
                .WithLeakRelu()
                .WithSgd()
                .Hidden()
                .Build();          

            var outputs = new LayerCreator(2, 2)
                .WithLearningRate(values[0])
                .WithMomentum(0)
                .FullSynapse()
                .Supervised()
                .DenseLayer()
                .WithSigmoid()
                .WithSgd()
                .Output()
                .Build();

            watch = System.Diagnostics.Stopwatch.StartNew();
            watch.Stop();
            Console.WriteLine($"Sinapse Time: {watch.ElapsedMilliseconds}ms");

            var (trainingValues, desiredValues) = Data.Get();

            int cont = 0;
            int sizeTrain = 10;

            var e = double.MaxValue;

            while (true)
            {
                watch = System.Diagnostics.Stopwatch.StartNew();
                e = 0;
                for (int i = 0; i < sizeTrain; i++)
                {
                    var index = i;
                    var inputs = trainingValues[index];
                    var desireds = desiredValues[index];

                    // Feed Forward
                    var _h = hiddens.Output(inputs);
                    var _o = outputs.Output(_h);

                    // Backward
                    var _oe = ((ISupervisedLearning)outputs).Learn(_h, desireds);
                    ((ISupervisedLearning)hiddens).Learn(inputs, _oe);

                    // Error
                    var e0 = Math.Abs(_o[0] - desireds[0]);
                    var e1 = Math.Abs(_o[1] - desireds[1]);
                    var error = Math.Sqrt(Math.Abs(e0 * e0 + e1 * e1));
                    e += error / 2.0;
                }

                e /= sizeTrain;
                cont++;
                watch.Stop();
                var time = watch.ElapsedMilliseconds;
                Console.WriteLine($"Interactions: {cont}\nError: {e}");
                Console.Title =
                    $"TSPS (Training Sample per Second): {Math.Ceiling(1000d / ((double)time / (double)sizeTrain))}";
            }
        }
    }
}
