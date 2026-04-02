import { existsSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

import { isEnvTruthy } from '../../utils/envUtils.js'

export const DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1'
export const DEFAULT_CODEX_BASE_URL = 'https://chatgpt.com/backend-api/codex'
/** Default GitHub Models API model when user selects copilot / github:copilot */
export const DEFAULT_GITHUB_MODELS_API_MODEL = 'openai/gpt-4.1'

/** Default Azure OpenAI API version — use the latest preview that supports the Responses API */
export const DEFAULT_AZURE_OPENAI_API_VERSION = '2025-03-01-preview'
/** Default Azure OpenAI model/deployment */
export const DEFAULT_AZURE_OPENAI_MODEL = 'gpt-5.4'

const CODEX_ALIAS_MODELS: Record<
  string,
  {
    model: string
    reasoningEffort?: ReasoningEffort
  }
> = {
  codexplan: {
    model: 'gpt-5.4',
    reasoningEffort: 'high',
  },
  'gpt-5.4': {
    model: 'gpt-5.4',
    reasoningEffort: 'high',
  },
  'gpt-5.3-codex': {
    model: 'gpt-5.3-codex',
    reasoningEffort: 'high',
  },
  'gpt-5.3-codex-spark': {
    model: 'gpt-5.3-codex-spark',
  },
  codexspark: {
    model: 'gpt-5.3-codex-spark',
  },
  'gpt-5.2-codex': {
    model: 'gpt-5.2-codex',
    reasoningEffort: 'high',
  },
  'gpt-5.1-codex-max': {
    model: 'gpt-5.1-codex-max',
    reasoningEffort: 'high',
  },
  'gpt-5.1-codex-mini': {
    model: 'gpt-5.1-codex-mini',
  },
  'gpt-5.4-mini': {
    model: 'gpt-5.4-mini',
    reasoningEffort: 'medium',
  },
  'gpt-5.2': {
    model: 'gpt-5.2',
    reasoningEffort: 'medium',
  },
} as const

type CodexAlias = keyof typeof CODEX_ALIAS_MODELS
type ReasoningEffort = 'low' | 'medium' | 'high' | 'xhigh'

export type ProviderTransport = 'chat_completions' | 'codex_responses' | 'azure_responses' | 'azure_chat_completions'

export type ResolvedProviderRequest = {
  transport: ProviderTransport
  requestedModel: string
  resolvedModel: string
  baseUrl: string
  reasoning?: {
    effort: ReasoningEffort
  }
}

export type ResolvedCodexCredentials = {
  apiKey: string
  accountId?: string
  authPath?: string
  source: 'env' | 'auth.json' | 'none'
}

type ModelDescriptor = {
  raw: string
  baseModel: string
  reasoning?: {
    effort: ReasoningEffort
  }
}

const LOCALHOST_HOSTNAMES = new Set(['localhost', '127.0.0.1', '::1'])

function asTrimmedString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function readNestedString(
  value: unknown,
  paths: string[][],
): string | undefined {
  for (const path of paths) {
    let current = value
    let valid = true
    for (const key of path) {
      if (!current || typeof current !== 'object' || !(key in current)) {
        valid = false
        break
      }
      current = (current as Record<string, unknown>)[key]
    }
    if (!valid) continue
    const stringValue = asTrimmedString(current)
    if (stringValue) return stringValue
  }
  return undefined
}

function decodeJwtPayload(token: string): Record<string, unknown> | undefined {
  const parts = token.split('.')
  if (parts.length < 2) return undefined

  try {
    const normalized = parts[1].replace(/-/g, '+').replace(/_/g, '/')
    const padded = normalized + '='.repeat((4 - (normalized.length % 4)) % 4)
    const json = Buffer.from(padded, 'base64').toString('utf8')
    const parsed = JSON.parse(json)
    return parsed && typeof parsed === 'object'
      ? (parsed as Record<string, unknown>)
      : undefined
  } catch {
    return undefined
  }
}

function parseReasoningEffort(value: string | undefined): ReasoningEffort | undefined {
  if (!value) return undefined
  const normalized = value.trim().toLowerCase()
  if (normalized === 'low' || normalized === 'medium' || normalized === 'high' || normalized === 'xhigh') {
    return normalized
  }
  return undefined
}

function parseModelDescriptor(model: string): ModelDescriptor {
  const trimmed = model.trim()
  const queryIndex = trimmed.indexOf('?')
  if (queryIndex === -1) {
    const alias = trimmed.toLowerCase() as CodexAlias
    const aliasConfig = CODEX_ALIAS_MODELS[alias]
    if (aliasConfig) {
      return {
        raw: trimmed,
        baseModel: aliasConfig.model,
        reasoning: aliasConfig.reasoningEffort
          ? { effort: aliasConfig.reasoningEffort }
          : undefined,
      }
    }
    return {
      raw: trimmed,
      baseModel: trimmed,
    }
  }

  const baseModel = trimmed.slice(0, queryIndex).trim()
  const params = new URLSearchParams(trimmed.slice(queryIndex + 1))
  const alias = baseModel.toLowerCase() as CodexAlias
  const aliasConfig = CODEX_ALIAS_MODELS[alias]
  const resolvedBaseModel = aliasConfig?.model ?? baseModel
  const reasoning =
    parseReasoningEffort(params.get('reasoning') ?? undefined) ??
    (aliasConfig?.reasoningEffort
      ? { effort: aliasConfig.reasoningEffort }
      : undefined)

  return {
    raw: trimmed,
    baseModel: resolvedBaseModel,
    reasoning: typeof reasoning === 'string' ? { effort: reasoning } : reasoning,
  }
}

function isCodexAlias(model: string): boolean {
  const normalized = model.trim().toLowerCase()
  const base = normalized.split('?', 1)[0] ?? normalized
  return base in CODEX_ALIAS_MODELS
}

export function isLocalProviderUrl(baseUrl: string | undefined): boolean {
  if (!baseUrl) return false
  try {
    return LOCALHOST_HOSTNAMES.has(new URL(baseUrl).hostname)
  } catch {
    return false
  }
}

export function isCodexBaseUrl(baseUrl: string | undefined): boolean {
  if (!baseUrl) return false
  try {
    const parsed = new URL(baseUrl)
    return (
      parsed.hostname === 'chatgpt.com' &&
      parsed.pathname.replace(/\/+$/, '') === '/backend-api/codex'
    )
  } catch {
    return false
  }
}

/**
 * Normalize user model string for GitHub Models inference (models.github.ai).
 * Mirrors runtime devsper `github._normalize_model_id`.
 */
export function normalizeGithubModelsApiModel(requestedModel: string): string {
  const noQuery = requestedModel.split('?', 1)[0] ?? requestedModel
  const segment =
    noQuery.includes(':') ? noQuery.split(':', 2)[1]!.trim() : noQuery.trim()
  if (!segment || segment.toLowerCase() === 'copilot') {
    return DEFAULT_GITHUB_MODELS_API_MODEL
  }
  return segment
}

export function resolveProviderRequest(options?: {
  model?: string
  baseUrl?: string
  fallbackModel?: string
  reasoningEffortOverride?: ReasoningEffort
}): ResolvedProviderRequest {
  const isGithubMode = isEnvTruthy(process.env.CLAUDE_CODE_USE_GITHUB)
  const requestedModel =
    options?.model?.trim() ||
    process.env.OPENAI_MODEL?.trim() ||
    options?.fallbackModel?.trim() ||
    (isGithubMode ? 'github:copilot' : 'gpt-4o')
  const descriptor = parseModelDescriptor(requestedModel)
  const rawBaseUrl =
    options?.baseUrl ??
    process.env.OPENAI_BASE_URL ??
    process.env.OPENAI_API_BASE ??
    undefined
  const transport: ProviderTransport =
    isCodexAlias(requestedModel) || isCodexBaseUrl(rawBaseUrl)
      ? 'codex_responses'
      : 'chat_completions'

  const resolvedModel =
    transport === 'chat_completions' &&
    isEnvTruthy(process.env.CLAUDE_CODE_USE_GITHUB)
      ? normalizeGithubModelsApiModel(requestedModel)
      : descriptor.baseModel

  const reasoning = options?.reasoningEffortOverride
    ? { effort: options.reasoningEffortOverride }
    : descriptor.reasoning


  return {
    transport,
    requestedModel,
    resolvedModel,
    baseUrl:
      (rawBaseUrl ??
        (transport === 'codex_responses'
          ? DEFAULT_CODEX_BASE_URL
          : DEFAULT_OPENAI_BASE_URL)
      ).replace(/\/+$/, ''),
    reasoning,
  }
}

export function resolveCodexAuthPath(
  env: NodeJS.ProcessEnv = process.env,
): string {
  const explicit = asTrimmedString(env.CODEX_AUTH_JSON_PATH)
  if (explicit) return explicit

  const codexHome = asTrimmedString(env.CODEX_HOME)
  if (codexHome) return join(codexHome, 'auth.json')

  return join(homedir(), '.codex', 'auth.json')
}

export function parseChatgptAccountId(
  token: string | undefined,
): string | undefined {
  if (!token) return undefined
  const payload = decodeJwtPayload(token)
  const fromClaim = asTrimmedString(
    payload?.['https://api.openai.com/auth.chatgpt_account_id'],
  )
  if (fromClaim) return fromClaim
  return asTrimmedString(payload?.chatgpt_account_id)
}

function loadCodexAuthJson(
  authPath: string,
): Record<string, unknown> | undefined {
  if (!existsSync(authPath)) return undefined
  try {
    const raw = readFileSync(authPath, 'utf8')
    const parsed = JSON.parse(raw)
    return parsed && typeof parsed === 'object'
      ? (parsed as Record<string, unknown>)
      : undefined
  } catch {
    return undefined
  }
}

export function resolveCodexApiCredentials(
  env: NodeJS.ProcessEnv = process.env,
): ResolvedCodexCredentials {
  const envApiKey = asTrimmedString(env.CODEX_API_KEY)
  const envAccountId =
    asTrimmedString(env.CODEX_ACCOUNT_ID) ??
    asTrimmedString(env.CHATGPT_ACCOUNT_ID)

  if (envApiKey) {
    return {
      apiKey: envApiKey,
      accountId: envAccountId ?? parseChatgptAccountId(envApiKey),
      source: 'env',
    }
  }

  const authPath = resolveCodexAuthPath(env)
  const authJson = loadCodexAuthJson(authPath)
  if (!authJson) {
    return {
      apiKey: '',
      authPath,
      source: 'none',
    }
  }

  const apiKey = readNestedString(authJson, [
    ['access_token'],
    ['accessToken'],
    ['tokens', 'access_token'],
    ['tokens', 'accessToken'],
    ['auth', 'access_token'],
    ['auth', 'accessToken'],
    ['token', 'access_token'],
    ['token', 'accessToken'],
    ['tokens', 'id_token'],
    ['tokens', 'idToken'],
  ])
  const accountId =
    envAccountId ??
    readNestedString(authJson, [
      ['account_id'],
      ['accountId'],
      ['tokens', 'account_id'],
      ['tokens', 'accountId'],
      ['auth', 'account_id'],
      ['auth', 'accountId'],
    ]) ??
    parseChatgptAccountId(apiKey)

  if (!apiKey) {
    return {
      apiKey: '',
      accountId,
      authPath,
      source: 'none',
    }
  }

  return {
    apiKey,
    accountId,
    authPath,
    source: 'auth.json',
  }
}

export function getReasoningEffortForModel(model: string): ReasoningEffort | undefined {
  const normalized = model.trim().toLowerCase()
  const base = normalized.split('?', 1)[0] ?? normalized
  const alias = base as CodexAlias
  const aliasConfig = CODEX_ALIAS_MODELS[alias]
  return aliasConfig?.reasoningEffort
}

// ---------------------------------------------------------------------------
// Azure OpenAI provider config
// ---------------------------------------------------------------------------

export type ResolvedAzureOpenAIRequest = {
  transport: 'azure_responses' | 'azure_chat_completions'
  endpoint: string
  deployment: string
  apiVersion: string
  model: string
  reasoning?: {
    effort: ReasoningEffort
  }
}

/**
 * Models that support the Azure OpenAI Responses API.
 * These models can use the /responses endpoint instead of /chat/completions.
 */
const AZURE_RESPONSES_API_MODELS = new Set([
  'gpt-5.4',
  'gpt-5.3-codex',
  'gpt-5.3-codex-spark',
  'gpt-5.2-codex',
  'gpt-5.2',
  'gpt-5.1-codex-max',
  'gpt-5.1-codex-mini',
  'gpt-4.1',
  'gpt-4.1-mini',
  'gpt-4.1-nano',
  'gpt-4o',
  'gpt-4o-mini',
])

/** Azure OpenAI models with recommended reasoning effort defaults */
const AZURE_MODEL_REASONING_DEFAULTS: Record<string, ReasoningEffort> = {
  'gpt-5.4': 'high',
  'gpt-5.3-codex': 'high',
  'gpt-5.2-codex': 'high',
  'gpt-5.1-codex-max': 'high',
  'gpt-5.2': 'medium',
}

export function isAzureOpenAIEndpoint(url: string | undefined): boolean {
  if (!url) return false
  return /openai\.azure\.com|cognitiveservices\.azure\.com|services\.ai\.azure\.com/.test(url)
}

export function resolveAzureOpenAIRequest(options?: {
  model?: string
  endpoint?: string
  deployment?: string
  apiVersion?: string
  useResponsesApi?: boolean
  reasoningEffortOverride?: ReasoningEffort
}): ResolvedAzureOpenAIRequest {
  const endpoint = (
    options?.endpoint ??
    process.env.AZURE_OPENAI_ENDPOINT ??
    ''
  ).replace(/\/+$/, '')

  const model =
    options?.model?.trim() ||
    process.env.AZURE_OPENAI_MODEL?.trim() ||
    process.env.OPENAI_MODEL?.trim() ||
    DEFAULT_AZURE_OPENAI_MODEL

  const deployment =
    options?.deployment?.trim() ||
    process.env.AZURE_OPENAI_DEPLOYMENT?.trim() ||
    model

  const apiVersion =
    options?.apiVersion?.trim() ||
    process.env.AZURE_OPENAI_API_VERSION?.trim() ||
    DEFAULT_AZURE_OPENAI_API_VERSION

  // Determine whether to use the Responses API
  const envUseResponses = process.env.AZURE_OPENAI_USE_RESPONSES_API
  const useResponses =
    options?.useResponsesApi ??
    (envUseResponses !== undefined
      ? envUseResponses.trim().toLowerCase() !== '0' &&
        envUseResponses.trim().toLowerCase() !== 'false'
      : AZURE_RESPONSES_API_MODELS.has(model.toLowerCase()))

  const transport: 'azure_responses' | 'azure_chat_completions' = useResponses
    ? 'azure_responses'
    : 'azure_chat_completions'

  // Resolve reasoning effort
  const descriptor = parseModelDescriptor(model)
  const reasoning = options?.reasoningEffortOverride
    ? { effort: options.reasoningEffortOverride }
    : descriptor.reasoning ??
      (AZURE_MODEL_REASONING_DEFAULTS[model.toLowerCase()]
        ? { effort: AZURE_MODEL_REASONING_DEFAULTS[model.toLowerCase()] }
        : undefined)

  return {
    transport,
    endpoint,
    deployment,
    apiVersion,
    model: descriptor.baseModel,
    reasoning,
  }
}
