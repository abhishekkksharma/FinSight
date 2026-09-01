"use client";

import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import {
  ChevronDown,
  Info,
  MoreHorizontal,
  Wallet,
  IndianRupee,
  CreditCard,
  ShieldCheck,
} from "lucide-react";

type FormState = {
  income: string;
  age: string;
  dependents: string;
  occupation: string;
  cityTier: string;
  monthlyEmi: string;
  tenureYears: string;
  minSavingsRate: string;
};

type PredictionResponse = {
  predicted_monthly_expense: number;
  affordability: {
    verdict: string;
    reason: string;
    income: number;
    monthly_emi: number;
    tenure_years: number;
    total_emi_commitment: number;
    current_surplus: number;
    surplus_after_emi: number;
    required_monthly_buffer: number;
    emi_to_income_ratio: number;
  };
};

const apiUrl =
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

const initialForm: FormState = {
  income: "75000",
  age: "25",
  dependents: "2",
  occupation: "Engineer",
  cityTier: "Tier-1",
  monthlyEmi: "15000",
  tenureYears: "5",
  minSavingsRate: "20",
};

const formatMoney = (value: number) =>
  new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 0,
    style: "currency",
    currency: "INR",
  }).format(value);

export default function App() {
  const [form, setForm] = useState<FormState>(initialForm);
  const [result, setResult] =
    useState<PredictionResponse | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const resultCards = useMemo(() => {
    if (!result) {
      return [
        {
          label: "Monthly expense",
          title: "Awaiting",
          subtitle:
            "Submit your details to see your predicted monthly expense.",
          icon: Wallet,
        },
        {
          label: "Affordability",
          title: "Ready",
          subtitle:
            "Check whether the proposed EMI comfortably fits your income.",
          icon: ShieldCheck,
        },
        {
          label: "After EMI",
          title: "Configurable",
          subtitle:
            "Your remaining monthly surplus will appear here.",
          icon: IndianRupee,
        },
        {
          label: "EMI commitment",
          title: "Standby",
          subtitle:
            "See the total financial commitment for your selected tenure.",
          icon: CreditCard,
        },
      ];
    }

    return [
      {
        label: "Predicted expense",
        title: formatMoney(result.predicted_monthly_expense),
        subtitle:
          "Your expected monthly spending based on the ML model.",
        icon: Wallet,
      },
      {
        label: "Affordability verdict",
        title: result.affordability.verdict,
        subtitle: result.affordability.reason,
        icon: ShieldCheck,
      },
      {
        label: "Surplus after EMI",
        title: formatMoney(
          result.affordability.surplus_after_emi
        ),
        subtitle:
          "The amount left each month after paying the EMI.",
        icon: IndianRupee,
      },
      {
        label: "Total EMI commitment",
        title: formatMoney(
          result.affordability.total_emi_commitment
        ),
        subtitle: `Total amount you will pay over ${result.affordability.tenure_years} years.`,
        icon: CreditCard,
      },
    ];
  }, [result]);

  const updateField = (
    field: keyof FormState,
    value: string
  ) => {
    setForm((current) => ({
      ...current,
      [field]: value,
    }));
  };

  const handleSubmit = async (
    event: FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();

    setError("");
    setIsLoading(true);

    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          income: Number(form.income),
          age: Number(form.age),
          dependents: Number(form.dependents),
          occupation: form.occupation,
          city_tier: form.cityTier,
          monthly_emi: Number(form.monthlyEmi),
          tenure_years: Number(form.tenureYears),
          min_savings_rate:
            Number(form.minSavingsRate) / 100,
        }),
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(
          payload.detail ?? "Prediction failed"
        );
      }

      setResult(payload);
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Unable to reach the prediction service"
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#f7f7f8] font-sans text-[#151515] sm:px-6 lg:px-10">
      {/* Main workspace */}
      <div className="mx-auto min-h-[calc(100vh-48px)] max-w-[1280px] overflow-hidden bg-[#f7f7f8] ">
        {/* ================= TOP NAV ================= */}
        <header className="flex h-[76px] items-center justify-between px-6 sm:px-8 lg:px-10">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-full bg-white shadow-sm">
              <div className="text-[25px] font-black tracking-[-4px]">
                F
              </div>
            </div>

            <div className="hidden text-[15px] font-semibold sm:block">
              Finsight
            </div>
          </div>
          
        </header>

        {/* ================= PAGE HEADING ================= */}
        <section className="relative overflow-hidden px-6 pb-0 pt-5 sm:px-10">
          {/* subtle blue glow similar to reference */}
          <div className="pointer-events-none absolute -top-28 left-1/2 h-64 w-[650px] -translate-x-1/2 rounded-full bg-[#a8c8f7] opacity-40 blur-[90px]" />

          <div className="relative">
            <div className="flex items-baseline gap-3 sm:gap-5">
              <h1 className="text-[36px] font-bold leading-none tracking-[-1.8px] sm:text-[40px]">
                Fix
              </h1>

              <span className="text-[36px] font-bold leading-none tracking-[-1.8px] text-[#c8c8c9] sm:text-[40px]">
                Your
              </span>

              <span className="text-[36px] font-bold leading-none tracking-[-1.8px] text-[#c8c8c9] sm:text-[40px]">
                Finances
              </span>
            </div>

            {/* Tabs */}
            <div className="mt-5 flex gap-7 border-b border-[#e7e7e8]">
              <button
                type="button"
                className="relative pb-3 text-[12px] font-semibold"
              >
                Affordability
                <span className="absolute bottom-[-1px] left-0 h-[2px] w-full bg-black" />
              </button>
            </div>
          </div>
        </section>

        {/* ================= CONTENT ================= */}
        <div className="px-5 pb-7 pt-7 sm:px-10 sm:pt-9">
          {/* ================= INPUT SECTION ================= */}
          <form onSubmit={handleSubmit} className="mb-5">
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {/* Personal details card */}
              <div className="rounded-[25px] bg-white p-6 shadow-[0_4px_18px_rgba(0,0,0,0.025)]">
                <div className="mb-5 flex items-start justify-between">
                  <div>
                    <h2 className="text-[18px] font-semibold">
                      Financial Profile
                    </h2>

                    <p className="mt-1 text-[11px] text-[#a0a0a0]">
                      Enter your personal and income details.
                    </p>
                  </div>

                </div>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <InputField
                    label="Income"
                    value={form.income}
                    type="number"
                    min="1"
                    onChange={(value) =>
                      updateField("income", value)
                    }
                  />

                  <InputField
                    label="Age"
                    value={form.age}
                    type="number"
                    min="18"
                    max="100"
                    onChange={(value) =>
                      updateField("age", value)
                    }
                  />

                  <InputField
                    label="Dependents"
                    value={form.dependents}
                    type="number"
                    min="0"
                    max="10"
                    onChange={(value) =>
                      updateField("dependents", value)
                    }
                  />

                  <SelectField
                    label="Occupation"
                    value={form.occupation}
                    options={[
                      "Engineer",
                      "Professional",
                      "Self_Employed",
                      "Student",
                      "Retired",
                    ]}
                    onChange={(value) =>
                      updateField("occupation", value)
                    }
                  />

                  <SelectField
                    label="City tier"
                    value={form.cityTier}
                    options={[
                      "Tier-1",
                      "Tier-2",
                      "Tier-3",
                    ]}
                    onChange={(value) =>
                      updateField("cityTier", value)
                    }
                  />
                </div>
              </div>

              {/* EMI details card */}
              <div className="rounded-[25px] bg-white p-6 shadow-[0_4px_18px_rgba(0,0,0,0.025)]">
                <div className="mb-5 flex items-start justify-between">
                  <div>
                    <h2 className="text-[18px] font-semibold">
                      EMI Planning
                    </h2>

                    <p className="mt-1 text-[11px] text-[#a0a0a0]">
                      Configure your proposed loan commitment.
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <InputField
                    label="Monthly EMI"
                    value={form.monthlyEmi}
                    type="number"
                    min="0"
                    onChange={(value) =>
                      updateField("monthlyEmi", value)
                    }
                  />

                  <InputField
                    label="Tenure (Years)"
                    value={form.tenureYears}
                    type="number"
                    min="1"
                    max="40"
                    onChange={(value) =>
                      updateField("tenureYears", value)
                    }
                  />

                  <InputField
                    label="Minimum buffer"
                    value={form.minSavingsRate}
                    type="number"
                    min="0"
                    max="90"
                    suffix="%"
                    onChange={(value) =>
                      updateField("minSavingsRate", value)
                    }
                  />

                  <div className="flex flex-col justify-end">
                    <div className="rounded-2xl bg-[#f7f7f8] px-4 py-3">
                      <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#a1a1a1]">
                        Current EMI ratio
                      </p>

                      <p className="mt-1 text-[18px] font-semibold">
                        {form.income && Number(form.income) > 0
                          ? `${(
                              (Number(form.monthlyEmi) /
                                Number(form.income)) *
                              100
                            ).toFixed(1)}%`
                          : "0%"}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Submit */}
            <div className="mt-5 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <button
                type="button"
                onClick={() => setForm(initialForm)}
                className="rounded-full px-5 py-3 text-xs font-medium text-[#777] transition hover:bg-white hover:text-black"
              >
                Reset
              </button>

              <button
                type="submit"
                className="rounded-full bg-black px-7 py-3 text-xs font-semibold text-white shadow-[0_5px_15px_rgba(0,0,0,0.12)] transition hover:bg-[#242424] active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50"
                disabled={isLoading}
              >
                {isLoading
                  ? "Checking..."
                  : "Check affordability"}
              </button>
            </div>

            {error && (
              <p className="mt-3 text-center text-xs font-medium text-red-500">
                {error}
              </p>
            )}
          </form>

          {/* ================= RESULT CARDS ================= */}
          <section
            className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4"
            aria-live="polite"
          >
            {resultCards.map((card) => {
              const Icon = card.icon;

              return (
                <article
                  key={card.label}
                  className="group relative flex min-h-[230px] flex-col overflow-hidden rounded-[25px] bg-white p-6 shadow-[0_4px_18px_rgba(0,0,0,0.025)] transition-all duration-300 hover:-translate-y-0.5 hover:shadow-[0_10px_30px_rgba(0,0,0,0.06)]"
                >
                  {/* Card heading */}
                  <div className="flex items-start justify-between">
                    <div>
                      <h2 className="text-[17px] font-semibold">
                        {card.label}
                      </h2>

                      <p className="mt-1 max-w-[210px] text-[11px] leading-relaxed text-[#a1a1a1]">
                        {card.label ===
                        "Predicted expense"
                          ? "Your estimated monthly spending."
                          : card.label ===
                              "Affordability verdict"
                            ? "Based on your current financial profile."
                            : card.label ===
                                "Surplus after EMI"
                              ? "Available after monthly obligations."
                              : "Loan commitment across the selected tenure."}
                      </p>
                    </div>

                    <button
                      type="button"
                      className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-[#e3e3e4] text-[#444] transition hover:bg-[#f7f7f7]"
                    >
                      <MoreHorizontal size={17} />
                    </button>
                  </div>

                  {/* Card content */}
                  <div className="mt-auto">
                    <div className="mb-4 flex items-center gap-3">
                      <div className="flex h-9 w-9 items-center justify-center rounded-full bg-[#f4f4f5]">
                        <Icon
                          size={16}
                          strokeWidth={1.8}
                        />
                      </div>

                      <div className="h-px flex-1 bg-[#eeeeef]" />
                    </div>

                    <div className="max-w-[240px]">
                      <h3 className="text-[21px] font-semibold leading-tight tracking-[-0.5px]">
                        {card.title}
                      </h3>

                      <p className="mt-2 line-clamp-3 text-[11px] leading-relaxed text-[#929292]">
                        {card.subtitle}
                      </p>
                    </div>
                  </div>

                  {/* Edit button */}
                  {/* <button
                    type="button"
                    className="absolute bottom-5 right-5 flex h-9 w-9 items-center justify-center rounded-full bg-black text-white transition-transform duration-200 group-hover:scale-105"
                    aria-label={`Edit ${card.label}`}
                  >
                    <Pencil size={14} />
                  </button> */}
                </article>
              );
            })}
          </section>

          {/* ================= BOTTOM ACTIONS ================= */}
          <div className="mt-5 flex items-center justify-between">
            <button
              type="button"
              className="flex h-9 w-9 items-center justify-center rounded-full bg-white shadow-sm transition hover:shadow-md"
              aria-label="Information"
            >
              <Info size={16} strokeWidth={1.8} />
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}

/* =========================================================
   REUSABLE INPUT COMPONENT
========================================================= */

type InputFieldProps = {
  label: string;
  value: string;
  type?: string;
  min?: string;
  max?: string;
  suffix?: string;
  onChange: (value: string) => void;
};

function InputField({
  label,
  value,
  type = "text",
  min,
  max,
  suffix,
  onChange,
}: InputFieldProps) {
  return (
    <label className="flex flex-col gap-2">
      <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#999]">
        {label}
      </span>

      <div className="relative">
        <input
          className={`w-full rounded-2xl border border-transparent bg-[#f5f5f6] px-4 py-3 text-[13px] font-medium text-[#181818] outline-none transition-all placeholder:text-[#aaa] focus:border-[#dedede] focus:bg-white ${
            suffix ? "pr-10" : ""
          }`}
          type={type}
          min={min}
          max={max}
          value={value}
          onChange={(event) =>
            onChange(event.target.value)
          }
        />

        {suffix && (
          <span className="absolute right-4 top-1/2 -translate-y-1/2 text-xs font-semibold text-[#999]">
            {suffix}
          </span>
        )}
      </div>
    </label>
  );
}

/* =========================================================
   REUSABLE SELECT COMPONENT
========================================================= */

type SelectFieldProps = {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
};

function SelectField({
  label,
  value,
  options,
  onChange,
}: SelectFieldProps) {
  return (
    <label className="flex flex-col gap-2">
      <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#999]">
        {label}
      </span>

      <div className="relative">
        <select
          className="w-full appearance-none rounded-2xl border border-transparent bg-[#f5f5f6] px-4 py-3 pr-10 text-[13px] font-medium text-[#181818] outline-none transition-all focus:border-[#dedede] focus:bg-white"
          value={value}
          onChange={(event) =>
            onChange(event.target.value)
          }
        >
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>

        <ChevronDown
          size={15}
          className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-[#777]"
        />
      </div>
    </label>
  );
}