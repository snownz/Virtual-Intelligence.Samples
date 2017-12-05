using System;
using VI.Neural.Factory;
using VI.Neural.Node;
using VI.NumSharp;

namespace DeepNetwork.MNIST
{
    class Program
    {
        static void Main(string[] args)
        {
            var rd = new Random();
            var values = new[] { .1f, .000f };

            var watch = System.Diagnostics.Stopwatch.StartNew();

            ProcessingDevice.Device = Device.CUDA;

            watch.Stop();
            Console.WriteLine($"Device Time: {watch.ElapsedMilliseconds}ms");

            var hiddens = new LayerCreator(30, 784)
                .WithLearningRate(values[0])
                .FullSynapse(.01f)
                .Supervised()
                .DenseLayer()
                .WithLeakRelu()
                .WithSgd()
                .Hidden()
                .Build();

            var hiddens2 = new LayerCreator(10, 30)
                .WithLearningRate(values[0])
                .FullSynapse(.01f)
                .Supervised()
                .DenseLayer()
                .WithLeakRelu()
                .WithSgd()
                .Hidden()
                .Build();

            var outputs = new LayerCreator(10, 10)
                .WithLearningRate(values[0])
                .FullSynapse(.01f)
                .Supervised()
                .SoftmaxLayer()
                .Nothing()
                .WithSgd()
                .Output()
                .Build();
            
            watch = System.Diagnostics.Stopwatch.StartNew();
            watch.Stop();
            Console.WriteLine($"Sinapse Time: {watch.ElapsedMilliseconds}ms");

            var trainingValues = MNISTLoader.OpenMnist();

            int cont = 0;
            int sizeTrain = trainingValues.Count;

            var e = double.MaxValue;

            while (true)
            {
                watch = System.Diagnostics.Stopwatch.StartNew();
                e = 0;
                for (int i = 0; i < sizeTrain; i++)
                {
                    var index = i; 

                    Console.WriteLine(trainingValues[index].ToString());
                    var inputs = ArrayMethods.ByteToArray(trainingValues[index].pixels, 28, 28);
                    var desireds = ArrayMethods.ByteToArray(trainingValues[index].label, 10);
                    
                    // Feed Forward
                    var _h = hiddens.Output(inputs);
                    var _h2 = hiddens2.Output(_h);
                    var _o = outputs.Output(_h2);

                    // Backward
                    var _oe = ((ISupervisedLearning)outputs).Learn(_h2, desireds);
                    var _he2 = ((ISupervisedLearning)hiddens2).Learn(_h, _oe);
                    ((ISupervisedLearning)hiddens).Learn(inputs, _he2);

                    ArrayMethods.PrintArray(_o, 10);
                    Console.WriteLine("\n");

                    // Error
                    var e0 = Math.Abs(_o[0] - desireds[0]);
                    var e1 = Math.Abs(_o[1] - desireds[1]);
                    var e2 = Math.Abs(_o[2] - desireds[2]);
                    var e3 = Math.Abs(_o[3] - desireds[3]);
                    var e4 = Math.Abs(_o[4] - desireds[4]);
                    var e5 = Math.Abs(_o[5] - desireds[5]);
                    var e6 = Math.Abs(_o[6] - desireds[6]);
                    var e7 = Math.Abs(_o[7] - desireds[7]);
                    var e8 = Math.Abs(_o[8] - desireds[8]);
                    var e9 = Math.Abs(_o[9] - desireds[9]);

                    var error
                        = Math.Sqrt(Math.Abs(e0 * e0 +
                                               e1 * e1 +
                                               e2 * e2 +
                                               e3 * e3 +
                                               e4 * e4 +
                                               e5 * e5 +
                                               e6 * e6 +
                                               e7 * e7 +
                                               e8 * e8 +
                                               e9 * e9
                                             ));
                    e += error / 2.0;
                }


                e /= sizeTrain;
                cont++;
                watch.Stop();
                var time = watch.ElapsedMilliseconds;
                Console.WriteLine($"Interactions: {cont}\nError: {e}");
                Console.Title =
                    $"Error: {e} --- TSPS (Training Sample per Second): {Math.Ceiling(1000d / ((double)time / (double)sizeTrain))}";
            }
        }
    }
}
