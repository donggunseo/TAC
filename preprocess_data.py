from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer
import random
import json
import argparse
import os

BANKING77_LABEL_LIST = ['card_arrival',
 'card_linking',
 'exchange_rate',
 'card_payment_wrong_exchange_rate',
 'extra_charge_on_statement',
 'pending_cash_withdrawal',
 'fiat_currency_support',
 'card_delivery_estimate',
 'automatic_top_up',
 'card_not_working',
 'exchange_via_app',
 'lost_or_stolen_card',
 'age_limit',
 'pin_blocked',
 'contactless_not_working',
 'top_up_by_bank_transfer_charge',
 'pending_top_up',
 'cancel_transfer',
 'top_up_limits',
 'wrong_amount_of_cash_received',
 'card_payment_fee_charged',
 'transfer_not_received_by_recipient',
 'supported_cards_and_currencies',
 'getting_virtual_card',
 'card_acceptance',
 'top_up_reverted',
 'balance_not_updated_after_cheque_or_cash_deposit',
 'card_payment_not_recognised',
 'edit_personal_details',
 'why_verify_identity',
 'unable_to_verify_identity',
 'get_physical_card',
 'visa_or_mastercard',
 'topping_up_by_card',
 'disposable_card_limits',
 'compromised_card',
 'atm_support',
 'direct_debit_payment_not_recognised',
 'passcode_forgotten',
 'declined_cash_withdrawal',
 'pending_card_payment',
 'lost_or_stolen_phone',
 'request_refund',
 'declined_transfer',
 'Refund_not_showing_up',
 'declined_card_payment',
 'pending_transfer',
 'terminate_account',
 'card_swallowed',
 'transaction_charged_twice',
 'verify_source_of_funds',
 'transfer_timing',
 'reverted_card_payment?',
 'change_pin',
 'beneficiary_not_allowed',
 'transfer_fee_charged',
 'receiving_money',
 'failed_transfer',
 'transfer_into_account',
 'verify_top_up',
 'getting_spare_card',
 'top_up_by_cash_or_cheque',
 'order_physical_card',
 'virtual_card_not_working',
 'wrong_exchange_rate_for_cash_withdrawal',
 'get_disposable_virtual_card',
 'top_up_failed',
 'balance_not_updated_after_bank_transfer',
 'cash_withdrawal_not_recognised',
 'exchange_charge',
 'top_up_by_card_charge',
 'activate_my_card',
 'cash_withdrawal_charge',
 'card_about_to_expire',
 'apple_pay_or_google_pay',
 'verify_my_identity',
 'country_support']
TREC_FINE_LABEL_LIST = [
    "abbreviation:abbreviation",     # 0
    "abbreviation:expansion",        # 1
    "entity:animal",                 # 2
    "entity:body",                   # 3
    "entity:color",                  # 4
    "entity:creation",               # 5
    "entity:currency",               # 6
    "entity:disease",                # 7
    "entity:event",                  # 8
    "entity:food",                   # 9
    "entity:instrument",             # 10
    "entity:language",               # 11
    "entity:letter",                 # 12
    "entity:other",                  # 13
    "entity:plant",                  # 14
    "entity:product",                # 15
    "entity:religion",               # 16
    "entity:sport",                  # 17
    "entity:substance",              # 18
    "entity:symbol",                 # 19
    "entity:method",                 # 20
    "entity:term",                   # 21
    "entity:vehicle",                # 22
    "entity:word",                   # 23
    "description:definition",         # 24
    "description:description",        # 25
    "description:manner",             # 26
    "description:reason",             # 27
    "human:group",                    # 28
    "human:individual",               # 29
    "human:title",                    # 30
    "human:description",              # 31
    "location:city",                  # 32
    "location:country",               # 33
    "location:mountain",              # 34
    "location:other",                 # 35
    "location:state",                 # 36
    "number:code",                    # 37
    "number:count",                   # 38
    "number:date",                    # 39
    "number:distance",                # 40
    "number:money",                   # 41
    "number:ordinal",                 # 42
    "number:other",                   # 43
    "number:period",                  # 44
    "number:percent",                 # 45
    "number:speed",                   # 46
    "number:temperature",             # 47
    "number:volume",                  # 48
    "number:weight",                  # 49
]
CLINC150_LABEL_LIST = [
    "kitchen_and_dining:restaurant_reviews",      # 0
    "kitchen_and_dining:nutrition_info",          # 1
    "banking:account_blocked",                    # 2
    "auto_and_commute:oil_change_how",            # 3
    "utility:time",                               # 4
    "utility:weather",                            # 5
    "credit_cards:redeem_rewards",                # 6
    "banking:interest_rate",                      # 7
    "auto_and_commute:gas_type",                  # 8
    "kitchen_and_dining:accept_reservations",     # 9
    "home:smart_home",                            # 10
    "meta:user_name",                             # 11
    "credit_cards:report_lost_card",              # 12
    "meta:repeat",                                # 13
    "meta:whisper_mode",                          # 14
    "small_talk:what_are_your_hobbies",           # 15
    "home:order",                                 # 16
    "auto_and_commute:jump_start",                # 17
    "work:schedule_meeting",                      # 18
    "work:meeting_schedule",                      # 19
    "banking:freeze_account",                     # 20
    "home:what_song",                             # 21
    "small_talk:meaning_of_life",                 # 22
    "kitchen_and_dining:restaurant_reservation",  # 23
    "auto_and_commute:traffic",                   # 24
    "utility:make_call",                          # 25
    "utility:text",                               # 26
    "banking:bill_balance",                       # 27
    "credit_cards:improve_credit_score",          # 28
    "meta:change_language",                       # 29
    "meta:no",                                    # 30
    "utility:measurement_conversion",             # 31
    "utility:timer",                              # 32
    "utility:flip_coin",                          # 33
    "small_talk:do_you_have_pets",                # 34
    "banking:balance",                            # 35
    "small_talk:tell_joke",                       # 36
    "auto_and_commute:last_maintenance",          # 37
    "travel:exchange_rate",                       # 38
    "auto_and_commute:uber",                      # 39
    "travel:car_rental",                          # 40
    "credit_cards:credit_limit",                  # 41
    "oos:oos",                                    # 42
    "home:shopping_list",                         # 43
    "credit_cards:expiration_date",               # 44
    "banking:routing",                            # 45
    "kitchen_and_dining:meal_suggestion",         # 46
    "auto_and_commute:tire_change",               # 47
    "home:todo_list",                             # 48
    "credit_cards:card_declined",                 # 49
    "credit_cards:rewards_balance",               # 50
    "meta:change_accent",                         # 51
    "travel:vaccines",                            # 52
    "home:reminder_update",                       # 53
    "kitchen_and_dining:food_last",               # 54
    "meta:change_ai_name",                        # 55
    "banking:bill_due",                           # 56
    "small_talk:who_do_you_work_for",             # 57
    "utility:share_location",                     # 58
    "travel:international_visa",                  # 59
    "home:calendar",                               # 60
    "travel:translate",                            # 61
    "travel:carry_on",                             # 62
    "travel:book_flight",                          # 63
    "work:insurance_change",                       # 64
    "home:todo_list_update",                       # 65
    "travel:timezone",                              # 66
    "kitchen_and_dining:cancel_reservation",       # 67
    "banking:transactions",                        # 68
    "credit_cards:credit_score",                   # 69
    "banking:report_fraud",                        # 70
    "banking:spending_history",                    # 71
    "auto_and_commute:directions",                 # 72
    "utility:spelling",                            # 73
    "work:insurance",                              # 74
    "small_talk:what_is_your_name",                # 75
    "home:reminder",                               # 76
    "small_talk:where_are_you_from",               # 77
    "auto_and_commute:distance",                   # 78
    "work:payday",                                 # 79
    "travel:flight_status",                        # 80
    "utility:find_phone",                          # 81
    "small_talk:greeting",                         # 82
    "utility:alarm",                               # 83
    "home:order_status",                           # 84
    "kitchen_and_dining:confirm_reservation",      # 85
    "kitchen_and_dining:cook_time",                # 86
    "credit_cards:damaged_card",                   # 87
    "meta:reset_settings",                         # 88
    "banking:pin_change",                          # 89
    "credit_cards:replacement_card_duration",      # 90
    "credit_cards:new_card",                       # 91
    "utility:roll_dice",                           # 92
    "work:income",                                 # 93
    "work:taxes",                                   # 94
    "utility:date",                                 # 95
    "small_talk:who_made_you",                      # 96
    "work:pto_request",                             # 97
    "auto_and_commute:tire_pressure",               # 98
    "small_talk:how_old_are_you",                   # 99
    "work:rollover_401k",                           # 100
    "work:pto_request_status",                      # 101
    "kitchen_and_dining:how_busy",                  # 102
    "credit_cards:application_status",              # 103
    "kitchen_and_dining:recipe",                    # 104
    "home:calendar_update",                         # 105
    "home:play_music",                              # 106
    "meta:yes",                                     # 107
    "work:direct_deposit",                          # 108
    "credit_cards:credit_limit_change",             # 109
    "auto_and_commute:gas",                         # 110
    "banking:pay_bill",                             # 111
    "kitchen_and_dining:ingredients_list",          # 112
    "travel:lost_luggage",                          # 113
    "small_talk:goodbye",                           # 114
    "small_talk:what_can_i_ask_you",                # 115
    "travel:book_hotel",                            # 116
    "small_talk:are_you_a_bot",                     # 117
    "home:next_song",                               # 118
    "meta:change_speed",                            # 119
    "travel:plug_type",                             # 120
    "meta:maybe",                                    # 121
    "work:w2",                                       # 122
    "auto_and_commute:oil_change_when",              # 123
    "small_talk:thank_you",                          # 124
    "home:shopping_list_update",                     # 125
    "work:pto_balance",                              # 126
    "banking:order_checks",                          # 127
    "travel:travel_alert",                           # 128
    "small_talk:fun_fact",                           # 129
    "meta:sync_device",                              # 130
    "auto_and_commute:schedule_maintenance",         # 131
    "credit_cards:apr",                              # 132
    "banking:transfer",                              # 133
    "kitchen_and_dining:ingredient_substitution",    # 134
    "kitchen_and_dining:calories",                   # 135
    "auto_and_commute:current_location",             # 136
    "credit_cards:international_fees",               # 137
    "utility:calculator",                            # 138
    "utility:definition",                            # 139
    "work:next_holiday",                             # 140
    "home:update_playlist",                          # 141
    "auto_and_commute:mpg",                          # 142
    "banking:min_payment",                           # 143
    "meta:change_user_name",                         # 144
    "kitchen_and_dining:restaurant_suggestion",      # 145
    "travel:travel_notification",                    # 146
    "meta:cancel",                                   # 147
    "work:pto_used",                                 # 148
    "travel:travel_suggestion",                      # 149
    "meta:change_volume"                             # 150
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="banking77")
    parser.add_argument("--save_dir", type=str, default="./dataset")
    args = parser.parse_args()


    os.makedirs(os.path.join(args.save_dir,args.dataset_name), exist_ok=True)
    ##For llama3.1 and Qwen2.5
    ##만약 모델이 달라지면 토큰수 고정 데이터셋 다시 만들어야함
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
    if "banking77" in args.dataset_name:
        dataset = load_dataset("mteb/banking77")
        class_label = ClassLabel(names = list(range(77)))
        dataset = dataset.cast_column('label', class_label)
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        if "3token" in args.dataset_name:
            filtered_label_list = []
            for l in BANKING77_LABEL_LIST:
                if len(tokenizer.tokenize(": "+l))-1==3:
                    filtered_label_list.append(l)
            with open(os.path.join(args.save_dir,args.dataset_name,"label_list.json"), "w") as f:
                json.dump(filtered_label_list, f)
            train_dataset = train_dataset.filter(lambda example:example['label_text'] in filtered_label_list)
            test_dataset = test_dataset.filter(lambda example:example['label_text'] in filtered_label_list)
        else:
            with open(os.path.join(args.save_dir,args.dataset_name,"label_list.json"), "w") as f:
                json.dump(BANKING77_LABEL_LIST, f)
        def transform(example):
            return {
                "input": example["text"],
                "output": example["label_text"]
            }
        ds = train_dataset.train_test_split(test_size=100, stratify_by_column = "label")
        train_dataset = ds['train']
        val_dataset = ds['test']
        train_dataset = train_dataset.map(transform, remove_columns=["text", "label", "label_text"])
        val_dataset = val_dataset.map(transform, remove_columns=["text", "label", "label_text"])
        test_dataset = test_dataset.map(transform, remove_columns=["text", "label", "label_text"])
    elif "trec" in args.dataset_name:
        dataset = load_dataset("CogComp/trec")
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        def convert_label(example):
            example["label_text"]=TREC_FINE_LABEL_LIST[example['fine_label']]
            return example
        train_dataset = train_dataset.map(convert_label)
        test_dataset = test_dataset.map(convert_label)
        if "3token" in args.dataset_name:
            filtered_label_list = []
            for l in TREC_FINE_LABEL_LIST:
                if len(tokenizer.tokenize(": "+l))-1==3:
                    filtered_label_list.append(l)
            with open(os.path.join(args.save_dir,args.dataset_name,"label_list.json"), "w") as f:
                json.dump(filtered_label_list, f)
            train_dataset = train_dataset.filter(lambda example:example['label_text'] in filtered_label_list)
            test_dataset = test_dataset.filter(lambda example:example['label_text'] in filtered_label_list)
        else:
            with open(os.path.join(args.save_dir,args.dataset_name,"label_list.json"), "w") as f:
                json.dump(TREC_FINE_LABEL_LIST, f)
        def transform(example):
            return {
                "input": example["text"],
                "output": example["label_text"]
            }
        ds = train_dataset.train_test_split(test_size=100, stratify_by_column = "fine_label")
        train_dataset = ds['train']
        val_dataset = ds['test']
        train_dataset = train_dataset.map(transform, remove_columns=["text", "coarse_label", "fine_label", "label_text"])
        val_dataset = val_dataset.map(transform, remove_columns=["text", "coarse_label", "fine_label", "label_text"])
        test_dataset = test_dataset.map(transform, remove_columns=["text", "coarse_label", "fine_label", "label_text"])
    elif "clinc" in args.dataset_name:
        dataset = load_dataset("clinc/clinc_oos", "small")
        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        test_dataset = dataset['test']
        def convert_label(example):
            example["label_text"]=CLINC150_LABEL_LIST[example['intent']]
            return example
        train_dataset = train_dataset.map(convert_label)
        val_dataset = val_dataset.map(convert_label)
        test_dataset = test_dataset.map(convert_label)
        CLINC150_LABEL_LIST.remove("oos:oos")
        with open(os.path.join(args.save_dir,args.dataset_name,"label_list.json"), "w") as f:
            json.dump(CLINC150_LABEL_LIST, f)
        train_dataset = train_dataset.filter(lambda example:example['label_text']!="oos:oos")
        val_dataset = val_dataset.filter(lambda example:example['label_text']!="oos:oos")
        test_dataset = test_dataset.filter(lambda example:example['label_text']!="oos:oos")
        def transform(example):
            return {
                "input": example["text"],
                "output": example["label_text"]
            }
        ds = val_dataset.train_test_split(test_size=300, stratify_by_column = "intent")
        val_dataset = ds['test']
        train_dataset = train_dataset.map(transform, remove_columns=["text", "intent", "label_text"])
        val_dataset = val_dataset.map(transform, remove_columns=["text", "intent", "label_text"])
        test_dataset = test_dataset.map(transform, remove_columns=["text", "intent", "label_text"])
    elif args.dataset_name == "wmt19_cs-en":
        dataset = load_dataset("wmt/wmt19", "cs-en")
        train_dataset = dataset['train']
        test_dataset = dataset['validation']
        ds = train_dataset.train_test_split(test_size = 300)
        train_dataset = ds['train']
        val_dataset = ds['test']
        train_dataset = train_dataset.shuffle(seed=42)
        train_dataset = train_dataset.select([i for i in range(5000)])
        def transform(example):
            return {
                "input": example["translation"]['cs'],
                "output": example["translation"]['en']
            }
        train_dataset = train_dataset.map(transform, remove_columns=["translation"])
        val_dataset = val_dataset.map(transform, remove_columns=["translation"])
        test_dataset = test_dataset.map(transform, remove_columns=["translation"])
    elif args.dataset_name == "wmt19_en-cs":
        dataset = load_dataset("wmt/wmt19", "cs-en")
        train_dataset = dataset['train']
        test_dataset = dataset['validation']
        ds = train_dataset.train_test_split(test_size = 300)
        train_dataset = ds['train']
        val_dataset = ds['test']
        train_dataset = train_dataset.shuffle(seed=42)
        train_dataset = train_dataset.select([i for i in range(5000)])
        def transform(example):
            return {
                "input": example["translation"]['en'],
                "output": example["translation"]['cs']
            }
        train_dataset = train_dataset.map(transform, remove_columns=["translation"])
        val_dataset = val_dataset.map(transform, remove_columns=["translation"])
        test_dataset = test_dataset.map(transform, remove_columns=["translation"])
    elif args.dataset_name == "wmt19_de-en":
        dataset = load_dataset("wmt/wmt19", "de-en")
        train_dataset = dataset['train']
        test_dataset = dataset['validation']
        ds = train_dataset.train_test_split(test_size = 300)
        train_dataset = ds['train']
        val_dataset = ds['test']
        train_dataset = train_dataset.shuffle(seed=42)
        train_dataset = train_dataset.select([i for i in range(5000)])
        def transform(example):
            return {
                "input": example["translation"]['de'],
                "output": example["translation"]['en']
            }
        train_dataset = train_dataset.map(transform, remove_columns=["translation"])
        val_dataset = val_dataset.map(transform, remove_columns=["translation"])
        test_dataset = test_dataset.map(transform, remove_columns=["translation"])
    elif args.dataset_name == "wmt19_en-de":
        dataset = load_dataset("wmt/wmt19", "de-en")
        train_dataset = dataset['train']
        test_dataset = dataset['validation']
        ds = train_dataset.train_test_split(test_size = 300)
        train_dataset = ds['train']
        val_dataset = ds['test']
        train_dataset = train_dataset.shuffle(seed=42)
        train_dataset = train_dataset.select([i for i in range(5000)])
        def transform(example):
            return {
                "input": example["translation"]['en'],
                "output": example["translation"]['de']
            }
        train_dataset = train_dataset.map(transform, remove_columns=["translation"])
        val_dataset = val_dataset.map(transform, remove_columns=["translation"])
        test_dataset = test_dataset.map(transform, remove_columns=["translation"])
    train_dataset = train_dataset.to_list()
    with open(os.path.join(args.save_dir,args.dataset_name, "train.json"), "w") as f:
        json.dump(train_dataset, f)
    val_dataset = val_dataset.to_list()
    with open(os.path.join(args.save_dir,args.dataset_name, "valid.json"), "w") as f:
        json.dump(val_dataset, f)
    test_dataset = test_dataset.to_list()
    with open(os.path.join(args.save_dir,args.dataset_name, "test.json"), "w") as f:
        json.dump(test_dataset, f)

        
            
