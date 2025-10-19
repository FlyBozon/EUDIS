import { useAppStore } from '@/store/index';
import type { Language } from '@/i18n/translations';

export const LanguageSwitch = () => {
  const language = useAppStore((state) => state.language);
  const setLanguage = useAppStore((state) => state.setLanguage);

  const toggleLanguage = () => {
    const newLanguage: Language = language === 'pl' ? 'en' : 'pl';
    setLanguage(newLanguage);
  };

  return (
    <button
      onClick={toggleLanguage}
      className="px-3 py-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 
                 rounded-lg shadow-md hover:shadow-lg transition-all duration-200 
                 border border-gray-200 dark:border-gray-700 font-medium text-sm"
      title={language === 'pl' ? 'Switch to English' : 'Przełącz na polski'}
    >
      {language === 'pl' ? 'PL' : 'EN'}
    </button>
  );
};