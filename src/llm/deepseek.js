/**
 * DeepSeek LLM 封装模块
 * 
 * 基于 DeepSeek 官方 OpenAI-compatible API 的封装
 * 官网文档: https://platform.deepseek.com/api-docs
 */

import dotenv from 'dotenv';
import OpenAI from 'openai';

// 加载 .env 文件中的环境变量
// 这样代码中可以通过 process.env.XXX 读取配置
dotenv.config();

/**
 * DeepSeek LLM 客户端类
 * 
 * 封装了与 DeepSeek API 的通信，提供简单的 chat() 方法
 * 支持功能:
 *   - 普通对话
 *   - 多轮对话（通过维护 messages 数组）
 *   - 工具调用（Function Calling）
 *   - 切换模型（deepseek-chat / deepseek-reasoner）
 */
export class DeepSeekLLM {
  /**
   * 创建 DeepSeekLLM 实例
   * 
   * @param {Object} options - 配置选项
   * @param {string} [options.apiKey] - DeepSeek API Key
   *                                 默认读取 DEEPSEEK_API_KEY 环境变量（从 .env 加载）
   * @param {string} [options.baseURL] - 自定义 API 地址
   *                                  默认: https://api.deepseek.com
   * @param {string} [options.model] - 默认模型名称
   *                                默认: deepseek-chat
   *                                可选: deepseek-reasoner (R1 推理模型)
   */
  constructor(options = {}) {
    // 设置默认模型，可在 chat() 调用时覆盖
    this.model = options.model || 'deepseek-chat';
    
    // 初始化 OpenAI 客户端
    // DeepSeek 提供 OpenAI-compatible API，所以可以直接用 OpenAI SDK
    this.client = new OpenAI({
      baseURL: options.baseURL || 'https://api.deepseek.com',
      apiKey: options.apiKey || process.env.DEEPSEEK_API_KEY,
    });
  }

  /**
   * 发送聊天请求到 DeepSeek API
   * 
   * 这是对外提供的核心方法，封装了底层的 chat.completions.create 调用
   * 
   * @param {Object} params - 请求参数
   * @param {Array<{role: string, content: string}>} params.messages - 消息列表
   *   role: 'system' | 'user' | 'assistant' | 'tool'
   *   content: 消息内容
   * @param {string} [params.model] - 模型名称，不传则使用构造时的默认模型
   * @param {number} [params.maxTokens] - 最大输出 token 数（控制回复长度）
   * @param {number} [params.temperature] - 采样温度 0-2（控制随机性，越低越确定）
   * @param {Array} [params.tools] - 工具定义列表，用于 Function Calling
   * 
   * @returns {Promise<ChatResponse>} 标准化响应对象
   * 
   * @example
   * // 最简单的调用
   * const response = await llm.chat({
   *   messages: [{ role: 'user', content: 'Hello!' }]
   * });
   * console.log(response.content);
   * 
   * @example
   * // 带工具调用的调用
   * const response = await llm.chat({
   *   messages: [{ role: 'user', content: '查北京天气' }],
   *   tools: [weatherTool]
   * });
   * if (response.hasToolCalls) {
   *   // 处理工具调用
   * }
   */
  async chat(params) {
    // 构建请求体，只包含必要的字段
    const requestBody = {
      model: params.model || this.model,
      messages: params.messages,
    };

    // 可选参数：只有显式传入时才添加到请求
    // 这样可以避免发送 undefined 值到 API
    if (params.maxTokens !== undefined) {
      requestBody.max_tokens = params.maxTokens;
    }

    if (params.temperature !== undefined) {
      requestBody.temperature = params.temperature;
    }

    // 工具调用支持
    // 传入 tools 数组时，让模型自动决定是否调用工具
    if (params.tools && params.tools.length > 0) {
      requestBody.tools = params.tools;
      requestBody.tool_choice = 'auto';  // auto = 模型自己决定
    }

    try {
      // 调用 DeepSeek API
      // 这个 API 与 OpenAI 完全一致，只是 baseURL 不同
      const completion = await this.client.chat.completions.create(requestBody);
      
      // 将原始响应转换为统一的格式
      return this._parseResponse(completion);
    } catch (error) {
      // 捕获网络错误、API 错误等，返回统一的错误格式
      // 这样调用方不需要 try-catch 也能正常处理错误
      return {
        content: `Error: ${error.message}`,
        finishReason: 'error',
        usage: {},
        toolCalls: [],
        hasToolCalls: false,
      };
    }
  }

  /**
   * 解析 OpenAI SDK 的响应为标准格式
   * 
   * @private
   * @param {Object} completion - OpenAI SDK 返回的原始响应
   * @returns {ChatResponse} 标准化的响应对象
   * 
   * 为什么要解析：
   * 1. 简化结构：只保留需要的字段
   * 2. 统一命名：将 snake_case 转为 camelCase
   * 3. 添加辅助字段：如 hasToolCalls 方便判断
   */
  _parseResponse(completion) {
    // OpenAI 响应结构：
    // completion.choices[0].message = { role, content, tool_calls? }
    const choice = completion.choices[0];
    const message = choice.message;

    return {
      // 回复内容，可能是 null（如果有工具调用）
      content: message.content || null,
      
      // 结束原因：
      // - 'stop': 正常完成
      // - 'length': 达到 max_tokens 限制
      // - 'tool_calls': 需要调用工具
      // - 'error': 发生错误
      finishReason: choice.finish_reason || 'stop',
      
      // Token 使用量统计
      usage: completion.usage ? {
        promptTokens: completion.usage.prompt_tokens,
        completionTokens: completion.usage.completion_tokens,
        totalTokens: completion.usage.total_tokens,
      } : {},
      
      // 工具调用列表，如果没有则为空数组
      toolCalls: message.tool_calls ? message.tool_calls.map(tc => ({
        id: tc.id,
        name: tc.function.name,
        arguments: tc.function.arguments,  // JSON 字符串，需要 JSON.parse
      })) : [],
      
      // 辅助字段：快速判断是否有工具调用
      hasToolCalls: !!(message.tool_calls && message.tool_calls.length > 0),
    };
  }
}

/**
 * ChatResponse 类型定义
 * 
 * 这是 JSDoc 的 @typedef 注释，用于 IDE 智能提示
 * 不会生成实际代码，但能让编辑器知道返回值的结构
 * 
 * 如果你使用 VSCode，鼠标悬停在 response 上可以看到这些字段说明
 * 
 * @typedef {Object} ChatResponse
 * @property {string|null} content - AI 的回复内容
 *                                    null 表示没有文本回复（如工具调用时）
 * @property {string} finishReason - 响应结束的原因
 *                                  'stop' | 'length' | 'error' | 'tool_calls'
 * @property {Object} usage - Token 使用量统计
 *                           @property {number} usage.promptTokens - 输入 token 数
 *                           @property {number} usage.completionTokens - 输出 token 数
 *                           @property {number} usage.totalTokens - 总计
 * @property {Array<{id: string, name: string, arguments: string}>} toolCalls - 
 *                            工具调用请求列表，arguments 是 JSON 字符串
 * @property {boolean} hasToolCalls - 是否有工具调用，方便快速判断
 *                                   if (response.hasToolCalls) { ... }
 */

// JSDoc @typedef 的作用：
// 1. IDE 智能提示：VSCode 等编辑器能识别这些类型，提供代码补全
// 2. 文档生成：可以生成 API 文档
// 3. 类型检查：配合 TypeScript 检查器可以进行类型检查
// 
// 如果你不需要这些功能，可以删除上面的 @typedef 注释块
// 代码运行时完全不受影响
