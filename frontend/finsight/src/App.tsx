import { useMemo, useState } from 'react'
import type { FormEvent } from 'react'

type FormState = {
  income: string
  age: string
  dependents: string
  occupation: string
  cityTier: string
  monthlyEmi: string
  tenureYears: string
  minSavingsRate: string
}

type PredictionResponse = {
  predicted_monthly_expense: number
  affordability: {
    verdict: string
    reason: string
    income: number
    monthly_emi: number
    tenure_years: number
    total_emi_commitment: number
    current_surplus: number
    surplus_after_emi: number
    required_monthly_buffer: number
    emi_to_income_ratio: number
  }
}

const apiUrl = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

const initialForm: FormState = {
  income: '75000',
  age: '25',
  dependents: '2',
  occupation: 'Engineer',
  cityTier: 'Tier-1',
  monthlyEmi: '15000',
  tenureYears: '5',
  minSavingsRate: '20',
}

const formatMoney = (value: number) =>
  new Intl.NumberFormat('en-IN', {
    maximumFractionDigits: 0,
    style: 'currency',
    currency: 'INR',
  }).format(value)

export default function App() {
  const [form, setForm] = useState<FormState>(initialForm)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const resultCards = useMemo(() => {
    if (!result) {
      return [
        { label: 'Current status', title: 'Awaiting', subtitle: 'Submit the form to see your predicted monthly expense and affordability.' },
        { label: 'Expense model', title: 'Ready', subtitle: 'The machine learning model is loaded and ready.' },
        { label: 'Buffer check', title: 'Configurable', subtitle: 'Adjust your savings buffer to see how it impacts the verdict.' },
        { label: 'Insights', title: 'Standby', subtitle: 'Actionable financial insights will appear here.' },
      ]
    }

    return [
      {
        label: 'Predicted expense',
        title: formatMoney(result.predicted_monthly_expense),
        subtitle: 'Your expected monthly spending based on the ML model.',
      },
      {
        label: 'Affordability verdict',
        title: result.affordability.verdict,
        subtitle: result.affordability.reason,
      },
      {
        label: 'Surplus after EMI',
        title: formatMoney(result.affordability.surplus_after_emi),
        subtitle: 'The amount you will have left each month after paying the EMI.',
      },
      {
        label: 'Total EMI commitment',
        title: formatMoney(result.affordability.total_emi_commitment),
        subtitle: `Total amount you will pay over ${result.affordability.tenure_years} years.`,
      },
    ]
  }, [result])

  const updateField = (field: keyof FormState, value: string) => {
    setForm((current) => ({ ...current, [field]: value }))
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError('')
    setIsLoading(true)

    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          income: Number(form.income),
          age: Number(form.age),
          dependents: Number(form.dependents),
          occupation: form.occupation,
          city_tier: form.cityTier,
          monthly_emi: Number(form.monthlyEmi),
          tenure_years: Number(form.tenureYears),
          min_savings_rate: Number(form.minSavingsRate) / 100,
        }),
      })

      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail ?? 'Prediction failed')
      }
      setResult(payload)
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : 'Unable to reach the prediction service',
      )
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-[#f3f4f6] text-gray-900 font-sans p-6 md:p-10">
      <nav className="flex items-center justify-between max-w-6xl mx-auto mb-16">
        <div className="flex items-center justify-center w-10 h-10 bg-black text-white rounded-full font-bold text-xl leading-none">
          f
        </div>
        <div className="flex items-center gap-4 text-sm font-medium">
          <a href="#planner" className="hover:text-gray-500 transition-colors hidden sm:block">Planner</a>
          <button className="px-4 py-2 bg-white hover:bg-gray-50 rounded-full border border-gray-200 transition-colors shadow-sm">
            Log in
          </button>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto flex flex-col items-center">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold tracking-tight mb-4 text-black">Check your affordability</h1>
          <p className="text-gray-500 text-sm max-w-md mx-auto leading-relaxed">
            Estimate your monthly expenses and see if a new EMI fits your income—gather predictions, buffer checks, or anything else that helps you plan.
          </p>
        </div>

        <div className="w-full max-w-3xl bg-white rounded-[2rem] p-8 shadow-sm border border-gray-100 mb-12">
          <form className="grid gap-6" onSubmit={handleSubmit}>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
              <label className="flex flex-col gap-1.5 text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
                Income
                <input
                  className="w-full bg-gray-50 border border-transparent focus:border-gray-200 focus:bg-white rounded-xl px-4 py-3 text-gray-900 text-sm outline-none transition-all shadow-sm"
                  min="1"
                  type="number"
                  value={form.income}
                  onChange={(event) => updateField('income', event.target.value)}
                />
              </label>
              <label className="flex flex-col gap-1.5 text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
                Age
                <input
                  className="w-full bg-gray-50 border border-transparent focus:border-gray-200 focus:bg-white rounded-xl px-4 py-3 text-gray-900 text-sm outline-none transition-all shadow-sm"
                  min="18"
                  max="100"
                  type="number"
                  value={form.age}
                  onChange={(event) => updateField('age', event.target.value)}
                />
              </label>
              <label className="flex flex-col gap-1.5 text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
                Dependents
                <input
                  className="w-full bg-gray-50 border border-transparent focus:border-gray-200 focus:bg-white rounded-xl px-4 py-3 text-gray-900 text-sm outline-none transition-all shadow-sm"
                  min="0"
                  max="10"
                  type="number"
                  value={form.dependents}
                  onChange={(event) => updateField('dependents', event.target.value)}
                />
              </label>
              <label className="flex flex-col gap-1.5 text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
                City tier
                <select
                  className="w-full bg-gray-50 border border-transparent focus:border-gray-200 focus:bg-white rounded-xl px-4 py-3 text-gray-900 text-sm outline-none transition-all shadow-sm appearance-none"
                  value={form.cityTier}
                  onChange={(event) => updateField('cityTier', event.target.value)}
                >
                  <option>Tier-1</option>
                  <option>Tier-2</option>
                  <option>Tier-3</option>
                </select>
              </label>
              <label className="flex flex-col gap-1.5 text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
                Occupation
                <select
                  className="w-full bg-gray-50 border border-transparent focus:border-gray-200 focus:bg-white rounded-xl px-4 py-3 text-gray-900 text-sm outline-none transition-all shadow-sm appearance-none"
                  value={form.occupation}
                  onChange={(event) => updateField('occupation', event.target.value)}
                >
                  <option>Engineer</option>
                  <option>Professional</option>
                  <option>Self_Employed</option>
                  <option>Student</option>
                  <option>Retired</option>
                </select>
              </label>
              <label className="flex flex-col gap-1.5 text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
                Monthly EMI
                <input
                  className="w-full bg-gray-50 border border-transparent focus:border-gray-200 focus:bg-white rounded-xl px-4 py-3 text-gray-900 text-sm outline-none transition-all shadow-sm"
                  min="0"
                  type="number"
                  value={form.monthlyEmi}
                  onChange={(event) => updateField('monthlyEmi', event.target.value)}
                />
              </label>
              <label className="flex flex-col gap-1.5 text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
                Tenure (Yrs)
                <input
                  className="w-full bg-gray-50 border border-transparent focus:border-gray-200 focus:bg-white rounded-xl px-4 py-3 text-gray-900 text-sm outline-none transition-all shadow-sm"
                  min="1"
                  max="40"
                  type="number"
                  value={form.tenureYears}
                  onChange={(event) => updateField('tenureYears', event.target.value)}
                />
              </label>
              <label className="flex flex-col gap-1.5 text-[11px] font-semibold text-gray-400 uppercase tracking-wider">
                Buffer %
                <input
                  className="w-full bg-gray-50 border border-transparent focus:border-gray-200 focus:bg-white rounded-xl px-4 py-3 text-gray-900 text-sm outline-none transition-all shadow-sm"
                  min="0"
                  max="90"
                  type="number"
                  value={form.minSavingsRate}
                  onChange={(event) => updateField('minSavingsRate', event.target.value)}
                />
              </label>
            </div>

            <div className="flex justify-center mt-4">
              <button
                className="px-8 py-3 bg-black hover:bg-gray-800 active:scale-95 text-white font-medium rounded-full transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md"
                disabled={isLoading}
                type="submit"
              >
                {isLoading ? 'Checking...' : 'Check affordability'}
              </button>
            </div>
            {error && <p className="text-red-500 text-xs font-semibold text-center mt-2">{error}</p>}
          </form>
        </div>

        <section className="w-full grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6" aria-live="polite">
          {resultCards.map((card) => (
            <article
              className="bg-white rounded-[2rem] p-6 flex flex-col min-h-[260px] shadow-sm border border-gray-100 hover:shadow-md transition-shadow group relative overflow-hidden"
              key={card.label}
            >
              <div className="flex flex-col z-10">
                <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-2">{card.label}</span>
                <h2 className="text-2xl font-bold text-black leading-tight">{card.title}</h2>
                <p className="text-sm text-gray-400 mt-4 leading-relaxed">{card.subtitle}</p>
              </div>
              <button className="absolute bottom-6 right-6 w-8 h-8 rounded-full border border-gray-200 flex items-center justify-center text-gray-400 group-hover:bg-gray-50 transition-colors">
                +
              </button>
            </article>
          ))}
        </section>
      </div>
    </main>
  )
}

