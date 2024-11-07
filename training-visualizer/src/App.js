import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

function App() {
  const [data, setData] = useState([]);
  const [stats, setStats] = useState({
    recentMean: 0,
    recentMin: 0,
    globalMin: 0,
    totalSteps: 0
  });

  // File upload handler
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      const text = await file.text();
      const lines = text.split('\n').filter(line => line.trim());
      const parsedData = lines.map(line => {
        try {
          return JSON.parse(line.replace(/'/g, '"'));
        } catch (e) {
          console.error('Error parsing line:', line, e);
          return null;
        }
      }).filter(item => item !== null);

      // Calculate statistics
      const recentData = parsedData.slice(-20);
      const recentMean = recentData.reduce((sum, item) => sum + item.loss, 0) / recentData.length;
      const recentMin = Math.min(...recentData.map(item => item.loss));
      const globalMin = Math.min(...parsedData.map(item => item.loss));

      setStats({
        recentMean,
        recentMin,
        globalMin,
        totalSteps: parsedData.length
      });

      // Calculate moving averages
      const windowSize = 20;
      const movingAverageData = parsedData.map((item, index) => {
        const window = parsedData.slice(Math.max(0, index - windowSize + 1), index + 1);
        const avgLoss = window.reduce((sum, curr) => sum + curr.loss, 0) / window.length;
        const avgGradNorm = window.reduce((sum, curr) => sum + curr.grad_norm, 0) / window.length;
        return {
          ...item,
          movingAvgLoss: avgLoss,
          movingAvgGradNorm: avgGradNorm
        };
      });

      setData(movingAverageData);
    }
  };

  return (
    <div className="p-4">
      <input
        type="file"
        onChange={handleFileUpload}
        className="mb-4 block w-full text-sm text-gray-500
          file:mr-4 file:py-2 file:px-4
          file:rounded-full file:border-0
          file:text-sm file:font-semibold
          file:bg-blue-50 file:text-blue-700
          hover:file:bg-blue-100"
      />

      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="text-sm mb-4 grid grid-cols-2 gap-4">
          <div>Recent Average Loss: {stats.recentMean.toFixed(4)}</div>
          <div>Recent Best Loss: {stats.recentMin.toFixed(4)}</div>
          <div>Global Best Loss: {stats.globalMin.toFixed(4)}</div>
          <div>Total Steps: {stats.totalSteps}</div>
        </div>

        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="epoch"
                label={{ value: 'Epoch', position: 'bottom' }}
              />
              <YAxis
                yAxisId="left"
                label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                label={{ value: 'Gradient Norm', angle: 90, position: 'insideRight' }}
              />
              <Tooltip />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="loss"
                stroke="#94a3b8"
                dot={false}
                opacity={0.3}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="movingAvgLoss"
                stroke="#2563eb"
                dot={false}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="movingAvgGradNorm"
                stroke="#16a34a"
                dot={false}
                opacity={0.8}
              />
              <ReferenceLine
                yAxisId="left"
                y={stats.globalMin}
                stroke="#dc2626"
                strokeDasharray="3 3"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default App;