/**
 * DeepSeek LLM 使用示例
 * 
 * 运行前设置环境变量:
 *   export DEEPSEEK_API_KEY="your-api-key"
 * 
 * 然后运行:
 *   node examples/demo.js
 */

import { DeepSeekLLM } from '../src/llm/index.js';

/**
 * 示例1: 最简单的对话 - 完全对应官网 demo
 */
async function simpleChat() {
  console.log('=== 1. Simple Chat (官网 Demo 等价写法) ===');
  
  // 对应官网: const openai = new OpenAI({...})
  const llm = new DeepSeekLLM({
    // apiKey 会自动读取 DEEPSEEK_API_KEY 环境变量
  });

  // 对应官网: await openai.chat.completions.create({...})
  const response = await llm.chat({
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Hello!' }
    ],
    model: 'deepseek-chat',
  });

  // 对应官网: console.log(completion.choices[0].message.content)
  console.log('Response:', response.content);
  console.log('Usage:', response.usage);
  console.log();
}

/**
 * 示例2: 多轮对话
 */
async function multiTurnChat() {
  console.log('=== 2. Multi-turn Chat ===');
  
  const llm = new DeepSeekLLM({});
  
  const messages = [
    { role: 'system', content: 'You are a helpful assistant.' }
  ];

  // 第一轮
  messages.push({ role: 'user', content: 'My name is Alice.' });
  let response = await llm.chat({ messages });
  
  console.log('User: My name is Alice.');
  console.log('Assistant:', response.content);
  
  messages.push({ role: 'assistant', content: response.content });

  // 第二轮
  messages.push({ role: 'user', content: 'What is my name?' });
  response = await llm.chat({ messages });
  
  console.log('User: What is my name?');
  console.log('Assistant:', response.content);
  console.log();
}

/**
 * 示例3: 工具调用 (Function Calling)
 */
async function toolCalling() {
  console.log('=== 3. Tool Calling ===');
  
  const llm = new DeepSeekLLM({});
  
  const tools = [
    {
      type: 'function',
      function: {
        name: 'get_weather',
        description: 'Get weather for a city',
        parameters: {
          type: 'object',
          properties: {
            city: { type: 'string', description: 'City name' },
            unit: { type: 'string', enum: ['celsius', 'fahrenheit'] }
          },
          required: ['city']
        }
      }
    }
  ];

  const response = await llm.chat({
    messages: [{ role: 'user', content: 'What\'s the weather in Beijing?' }],
    tools: tools,
  });

  if (response.hasToolCalls) {
    console.log('Tool calls requested:');
    for (const tc of response.toolCalls) {
      console.log(`  - ${tc.name}(${tc.arguments})`);
    }
  } else {
    console.log('Response:', response.content);
  }
  console.log();
}

/**
 * 示例4: 使用 R1 推理模型
 */
async function reasoningModel() {
  console.log('=== 4. Reasoning Model (R1) ===');
  
  const llm = new DeepSeekLLM({});
  
  const response = await llm.chat({
    messages: [{ role: 'user', content: 'Solve: 23 * 47 + 89' }],
    model: 'deepseek-reasoner',  // R1 推理模型
  });

  console.log('Response:', response.content);
  console.log('Finish reason:', response.finishReason);
  console.log();
}

/**
 * 主函数
 */
async function main() {
  // .env 文件会自动加载，无需手动设置环境变量
  // 如果未找到 API Key，会给出提示
  if (!process.env.DEEPSEEK_API_KEY) {
    console.error('Error: DEEPSEEK_API_KEY not found');
    console.error('Please create a .env file with:');
    console.error('  DEEPSEEK_API_KEY="your-api-key"');
    process.exit(1);
  }

  try {
    await simpleChat();
    await multiTurnChat();
    await toolCalling();
    await reasoningModel();
  } catch (error) {
    console.error('Error:', error.message);
  }
}

main();
