using System;
using VI.NumSharp.Arrays;

namespace DeepNetwork.MNIST
{
    public static class ArrayMethods
    {
        public static float[] ByteToArray(byte b, int range)
        {
            var f = Convert.ToInt32(b);
            var br = new float[range];
            br[f] = 1;
            return br;
        }

        public static void PrintArray(float[] b, int range)
        {
            var str = "[";
            for (int i = 0; i < range - 1; i++)
            {
                str += $"{Math.Round(b[i], 2)}, ";
            }
            str += Math.Round(b[range - 1], 2) + "] = " + ArrayToInt(b, range);
            Console.WriteLine(str);
        }

        public static void PrintArray(Array<float> b, int range)
        {
            var str = "[";
            for (int i = 0; i < range - 1; i++)
            {
                str += $"{b[i]}, ";
            }
            str += b[range - 1] + "] = " + ArrayToInt(b, range);
            Console.WriteLine(str);
        }

        public static int ArrayToInt(Array<float> b, int range)
        {
            int r = 0;
            float max = 0;
            for (int i = 0; i < range; i++)
            {
                if (b[i] > max)
                {
                    max = b[i];
                    r = i;
                }
            }
            return r;
        }

        public static int ArrayToInt(float[] b, int range)
        {
            int r = 0;
            float max = 0;
            for (int i = 0; i < range; i++)
            {
                if (b[i] > max)
                {
                    max = b[i];
                    r = i;
                }
            }
            return r;
        }

        public static float[] ByteToArray(byte[][] b, int rangeX, int rangeY)
        {
            var f = new float[rangeX * rangeY];

            for (int x = 0; x < rangeX; x++)
            {
                for (int y = 0; y < rangeY; y++)
                {
                    f[x + y * rangeX] = (b[x][y] / 255f);
                }
            }
            return f;
        }          
    }
}
